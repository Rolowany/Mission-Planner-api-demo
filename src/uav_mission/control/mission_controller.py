import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from pymavlink import mavutil

from src.uav_mission.avoidance.avoider import Decision, DecisionKind, ObstacleAvoider
from src.uav_mission.geo import GeoPoint, to_global, to_local
from src.uav_mission.sensors.lidar import Lidar, SensorError
from src.uav_mission.utils.errors import MissionError

log = logging.getLogger("mission")
mav = mavutil.mavlink

EARTH_RADIUS_M = 6_378_137.0
# Sterujemy tylko pozycją: ignorujemy prędkość, przyspieszenie, yaw i yaw rate.
POSITION_ONLY_MASK = 0b0000_1111_1111_1000


@dataclass(frozen=True)
class Waypoint:
    id: str
    position: GeoPoint
    ordered: bool


@dataclass(frozen=True)
class ControllerConfig:
    altitude_m: float = 15.0
    accept_radius_m: float = 2.0
    # Prędkość przelotu. Domyślne 10 m/s ArduCoptera daje ok. 20 m drogi
    # hamowania, więcej niż zasięg wykrywania przeszkód.
    cruise_speed_mps: float = 5.0
    takeoff_timeout_s: float = 60.0
    # Limit na odcinek: stała część + czas przelotu z połową prędkości przelotowej.
    leg_base_timeout_s: float = 20.0
    land_timeout_s: float = 300.0
    max_detour_depth: int = 3  # objazd w objeździe w objeździe...


class MissionController:
    def __init__(self, master, waypoints: Sequence[Waypoint], home: GeoPoint,
                 config: ControllerConfig = ControllerConfig(),
                 lidar: Lidar | None = None, avoider: ObstacleAvoider | None = None):
        """`home` to początek lokalnego układu metrycznego, w którym pracują
        LiDAR i logika omijania (miejsce startu drona). Bez `lidar` dron leci
        bez omijania przeszkód."""
        if (lidar is None) != (avoider is None):
            raise ValueError("lidar i avoider podaje się razem albo wcale")
        self.master = master
        self.waypoints = list(waypoints)
        self.home = home
        self.config = config
        self.lidar = lidar
        self.avoider = avoider
        self.detours = 0

    def is_autopilot_heartbeat(self, msg) -> bool:
        return (msg is not None
                and msg.get_type() == "HEARTBEAT"
                and msg.get_srcSystem() == self.master.target_system
                and msg.get_srcComponent() == self.master.target_component)

    def send_command(self, command: int, *params: float, timeout: float = 5) -> bool:
        """Wysyła COMMAND_LONG i czeka na COMMAND_ACK. Zwraca True, gdy zaakceptowano."""
        return send_command(self.master, command, *params, timeout=timeout)

    def get_position(self, timeout: float = 3):
        """Zwraca (lat, lon, wysokość względna w m)."""
        return read_position(self.master, timeout)

    def set_mode(self, mode: str, timeout: float = 10) -> None:
        mapping = mavutil.mode_mapping_byname(self.master.vehicle_type) or {}
        if mode not in mapping:
            raise MissionError(f"Nieznany tryb: {mode}")
        mode_id = mapping[mode]
        self.master.set_mode(mode_id)

        deadline = time.time() + timeout
        while time.time() < deadline:
            hb = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if self.is_autopilot_heartbeat(hb) and hb.custom_mode == mode_id:
                log.info("Tryb: %s", mode)
                return
        raise MissionError(f"Nie udało się przełączyć w tryb {mode}")

    def arm(self, timeout: float = 60) -> None:
        """Uzbraja silniki. Ponawia próbę, bo zaraz po starcie SITL
        kontrole przed lotem (GPS, EKF) mogą jeszcze nie przechodzić."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.send_command(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1):
                log.info("Silniki uzbrojone")
                return
            log.info("Uzbrojenie odrzucone (kontrole przed lotem?), ponawiam za 2 s")
            time.sleep(2)
        raise MissionError("Nie udało się uzbroić silników")

    def takeoff(self) -> None:
        alt_m = self.config.altitude_m
        if not self.send_command(mav.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt_m):
            raise MissionError("Komenda startu odrzucona")

        deadline = time.time() + self.config.takeoff_timeout_s
        while time.time() < deadline:
            _, _, alt = self.get_position()
            if alt >= alt_m * 0.95:
                log.info("Osiągnięto wysokość %.1f m", alt)
                return
        raise MissionError("Przekroczono czas wznoszenia")

    def set_cruise_speed(self) -> None:
        speed = self.config.cruise_speed_mps
        for name, value in (("WP_SPD", speed), ("WPNAV_SPEED", speed * 100)):
            if set_param(self.master, name, value, timeout=2):
                log.info("Prędkość przelotowa: %.1f m/s (%s=%g)", speed, name, value)
                return
            log.info("Autopilot nie zna parametru %s, próbuję starszej nazwy", name)
        raise MissionError("Nie udało się ustawić prędkości przelotowej "
                           "(ani WP_SPD, ani WPNAV_SPEED)")

    def fly_to(self, target: GeoPoint, depth: int = 0) -> None:
        """Leci do celu. Jeśli jest LiDAR, w każdej iteracji sprawdza drogę
        i w razie potrzeby wykonuje objazd (rekurencyjnie, z limitem głębokości)."""
        self._send_position_target(target)
        self._reset_avoider()
        deadline = self._leg_deadline(target)
        braked = False

        while time.time() < deadline:
            lat, lon, alt = self.get_position()
            here = GeoPoint(lat, lon)
            if distance_m(lat, lon, target.lat, target.lon) <= self.config.accept_radius_m:
                return

            decision = self._check_obstacles(here, alt, target)
            if decision is None or decision.kind is DecisionKind.CLEAR:
                if braked:  # po zatrzymaniu przeszkoda zniknęła: fałszywy alarm
                    log.info("Po zatrzymaniu droga wolna - lecę dalej")
                    self._send_position_target(target)
                    braked = False
                continue

            if not braked:
                # Najpierw stop, potem decyzja z miejsca postoju: objazd
                # wyznaczony w ruchu wymagałby ostrego skrętu z pełną prędkością.
                log.warning("Przeszkoda: %s - hamuję", decision.reason)
                self._brake(here)
                braked = True
                deadline = self._leg_deadline(target)
                continue

            if decision.kind is DecisionKind.NO_PATH:
                self._hold(here)
                raise MissionError(f"Nie da się ominąć przeszkody: {decision.reason}")

            if depth >= self.config.max_detour_depth:
                self._hold(here)
                raise MissionError("Zbyt wiele kolejnych objazdów - przerywam")
            self.detours += 1
            detour = to_global(self.home, decision.detour)
            log.warning("Przeszkoda! %s -> punkt objazdu %.7f, %.7f",
                        decision.reason, detour.lat, detour.lon)
            self.fly_to(detour, depth + 1)
            log.info("Objazd zakończony, wracam na kurs")
            self._send_position_target(target)
            self._reset_avoider()
            braked = False
            deadline = self._leg_deadline(target)

        raise MissionError("Nie osiągnięto punktu w wyznaczonym czasie")

    def wait_disarmed(self) -> None:
        deadline = time.time() + self.config.land_timeout_s
        while time.time() < deadline:
            hb = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if self.is_autopilot_heartbeat(hb) and not (
                    hb.base_mode & mav.MAV_MODE_FLAG_SAFETY_ARMED):
                log.info("Wylądowano, silniki rozbrojone")
                return
        raise MissionError("Dron nie wylądował w wyznaczonym czasie")

    def run_mission(self) -> None:
        self.set_mode("GUIDED")
        self.arm()
        self.takeoff()
        self.set_cruise_speed()

        for i, wp in enumerate(self.waypoints, start=1):
            log.info("Punkt %d/%d: %s", i, len(self.waypoints), wp.id)
            self.fly_to(wp.position)
            log.info("Zaliczono %s", wp.id)

        self.set_mode("RTL")
        self.wait_disarmed()

    def emergency_rtl(self) -> None:
        try:
            log.warning("Awaryjny powrót (RTL)")
            self.set_mode("RTL")
        except MissionError as exc:
            log.error("RTL nieudany: %s - przejmij sterowanie ręcznie!", exc)

    def _send_position_target(self, target: GeoPoint) -> None:
        self.master.mav.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mav.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, POSITION_ONLY_MASK,
            int(round(target.lat * 1e7)), int(round(target.lon * 1e7)), self.config.altitude_m,
            0, 0, 0,   # prędkość (ignorowana)
            0, 0, 0,   # przyspieszenie (ignorowane)
            0, 0)      # yaw, yaw rate (ignorowane)

    def _hold(self, here: GeoPoint) -> None:
        """Zatrzymanie: nowym celem staje się bieżąca pozycja."""
        log.warning("Zatrzymuję drona w miejscu")
        self._send_position_target(here)

    def _brake(self, here: GeoPoint, max_speed_mps: float = 0.3, timeout: float = 10.0) -> None:
        """Zatrzymuje drona i czeka, aż prędkość pozioma spadnie prawie do zera."""
        self._hold(here)
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
            if msg is not None and math.hypot(msg.vx, msg.vy) / 100.0 < max_speed_mps:
                return
        raise MissionError("Dron nie zatrzymał się przed przeszkodą")

    def _check_obstacles(self, here: GeoPoint, alt: float, target: GeoPoint) -> Decision | None:
        if self.lidar is None:
            return None
        position = to_local(self.home, here)
        try:
            scan = self.lidar.scan(position, alt)
        except SensorError as exc:
            # Lot na ślepo jest niedopuszczalny: stop, a decyzję (RTL) podejmuje main.
            self._hold(here)
            raise MissionError(f"Awaria LiDAR-u: {exc}") from exc
        return self.avoider.check(scan, position, to_local(self.home, target))

    def _reset_avoider(self) -> None:
        if self.avoider is not None:
            self.avoider.reset()

    def _leg_deadline(self, target: GeoPoint) -> float:
        lat, lon, _ = self.get_position()
        distance = distance_m(lat, lon, target.lat, target.lon)
        return (time.time() + self.config.leg_base_timeout_s
                + distance / (0.5 * self.config.cruise_speed_mps))

def send_command(master, command: int, *params: float, timeout: float = 5) -> bool:
    """Wysyła COMMAND_LONG i czeka na COMMAND_ACK. Zwraca True, gdy zaakceptowano."""
    padded = list(params) + [0.0] * (7 - len(params))
    master.mav.command_long_send(
        master.target_system, master.target_component, command, 0, *padded)

    deadline = time.time() + timeout
    while time.time() < deadline:
        ack = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
        if ack and ack.command == command:
            if ack.result != mav.MAV_RESULT_ACCEPTED:
                log.warning("Komenda %d odrzucona (wynik %d)", command, ack.result)
            return ack.result == mav.MAV_RESULT_ACCEPTED
    log.warning("Brak potwierdzenia komendy %d", command)
    return False


def set_param(master, name: str, value: float, timeout: float = 5) -> bool:
    master.mav.param_set_send(master.target_system, master.target_component,
                              name.encode(), value, mav.MAV_PARAM_TYPE_REAL32)
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if msg is not None and msg.param_id == name:
            if abs(msg.param_value - value) > 1e-3 * max(1.0, abs(value)):
                raise MissionError(f"Autopilot ustawił {name}={msg.param_value} zamiast {value}")
            return True
    return False


def read_position(master, timeout: float = 3):
    """Zwraca (lat, lon, wysokość względna w m)."""
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout)
    if msg is None:
        raise MissionError("Brak danych o pozycji (GLOBAL_POSITION_INT)")
    return msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0


def wait_for_gps(master, timeout: float = 60):
    """Czeka na pierwszą sensowną pozycję (zaraz po starcie SITL bywa 0, 0)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        lat, lon, alt = read_position(master)
        if lat != 0 or lon != 0:
            return lat, lon, alt
    raise MissionError("Brak pozycji GPS")


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Odległość pozioma w metrach (przybliżenie równoodległościowe)."""
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return math.hypot(x, y) * EARTH_RADIUS_M


def connect(url: str, timeout: float = 30):
    """Łączy się z autopilotem, ustawia jego ID jako cel komend
    i zamawia strumień pozycji."""
    log.info("Łączenie z %s ...", url)
    master = mavutil.mavlink_connection(url)

    deadline = time.time() + timeout
    while time.time() < deadline:
        hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        # Pomijamy heartbeaty stacji naziemnych (np. Mission Plannera),
        # które autopilot może przekazywać dalej.
        if hb and hb.type != mav.MAV_TYPE_GCS:
            master.target_system = hb.get_srcSystem()
            master.target_component = hb.get_srcComponent()
            master.vehicle_type = hb.type
            log.info("Połączono z systemem %d, komponent %d",
                     master.target_system, master.target_component)
            if not send_command(master, mav.MAV_CMD_SET_MESSAGE_INTERVAL,
                                mav.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 200_000):  # 5 Hz
                raise MissionError("Autopilot nie chce wysyłać pozycji")
            return master
    raise MissionError("Brak heartbeatu autopilota - czy SITL jest uruchomiony?")