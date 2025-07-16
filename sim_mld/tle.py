import math
from datetime import datetime
from skyfield.api import load, EarthSatellite, wgs84
from sgp4.api import SatrecArray
import numpy as np

def load_url_tle_data(url: str, reload: bool = True) -> tuple:
    """
    Load Starlink TLE data from the provided URL.

    Parameters:
        url (str): URL to the TLE file.
        reload (bool): Whether to reload the TLE data.

    Returns:
        tuple: (timescale, valid_satellites, sat_array)
    """
    ts = load.timescale()
    satellites = load.tle_file(url, reload=reload)
    if not satellites:
        raise Exception("No Starlink satellites were loaded; check the TLE URL.")
    print("Loaded %d Starlink satellites from %s" % (len(satellites), url))

    valid_satellites = []
    for sat in satellites:
        pos = sat.at(ts.now())
        if np.isnan(pos.position.km).any():
            message = pos.message if pos.message else "position is invalid"
            print("Skipping %s due to error: %s" % (sat.name, message))
            continue
        valid_satellites.append(sat)

    models = [sat.model for sat in valid_satellites]
    sat_array = SatrecArray(models)
    return ts, valid_satellites, sat_array

def tle_checksum(line: str) -> int:
    """
    Compute the TLE checksum by summing all digits and treating each '-' as 1.
    The checksum is the sum modulo 10.
    """
    checksum = 0
    for char in line:
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    return checksum % 10

def format_line(line: str) -> str:
    """Append the computed checksum at the end of a TLE line."""
    return line + str(tle_checksum(line))

def format_epoch(dt: datetime) -> str:
    """
    Convert a datetime to the TLE epoch format: YYDDD.FFFF,
    where YY is the last two digits of the year, DDD is the day of year,
    and FFFF is the fraction of the day.
    """
    year = dt.year % 100
    day_of_year = dt.timetuple().tm_yday
    seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    frac_day = seconds_since_midnight / 86400
    return f"{year:02d}{day_of_year:03d}.{frac_day:06.4f}"

def generate_tle(sat_num, epoch_str, inclination, raan, eccentricity,
                 arg_perigee, mean_anomaly, mean_motion, rev_num):
    """
    Generate a synthetic TLE (satellite name and two standard lines) using the provided orbital elements.
    Note: This is for simulation purposes only and is not valid for real-world orbit propagation.
    """
    # Satellite name (Line 0)
    line0 = f"SAT-{sat_num:05d}"
    
    # Line 1: Basic catalog info using a fixed international designator.
    line1_no_cs = (f"1 {sat_num:05d}U 00000A   {epoch_str}  .00000000  00000-0  00000-0 0")
    line1 = format_line(line1_no_cs)
    
    # TLE requires eccentricity without a leading decimal point.
    ecc_str = f"{eccentricity:.7f}".split('.')[1]  # for example, 0.001 becomes '0010000'
    line2_no_cs = (f"2 {sat_num:05d} {inclination:8.4f} {raan:8.4f} {ecc_str:7s} "
                   f"{arg_perigee:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f}{rev_num:5d}")
    line2 = format_line(line2_no_cs)
    
    return line0, line1, line2


def fetch_valid_sat_from_tle_list(tle_list):
    ts = load.timescale()
    valid_satellites = []
    for sat_name, line1, line2 in tle_list:
        # Create the EarthSatellite object using Skyfield.
        sat = EarthSatellite(line1, line2, sat_name, ts)
        pos = sat.at(ts.now())
        if np.isnan(pos.position.km).any():
            message = pos.message if pos.message else "position is invalid"
            print("Skipping %s due to error: %s" % (sat.name, message))
            continue
        valid_satellites.append(sat)

    models = [sat.model for sat in valid_satellites]
    sat_array = SatrecArray(models)
    return ts, valid_satellites, sat_array

def generate_walker_constellation(sats_per_plane = 50, planes = 20, phasing = 1, epoch = datetime(2025, 1, 1),
                                  inclination=50, eccentricity=0.0001, arg_perigee = 0., mean_motion=15.0):
    """
    Generate a Walker Delta constellation by assigning RAAN and mean anomaly for each satellite.
    Satellites are distributed evenly across the planes and within each plane.
    """
    total_sats = sats_per_plane * planes
    epoch_str = format_epoch(epoch)
    tle_list = []
    
    for i in range(total_sats):
        # Determine which orbital plane and the satellite's slot in that plane.
        plane = i % planes         # Plane index: 0 to (planes - 1)
        sat_in_plane = i // planes   # Position within the plane
        
        # RAAN is evenly spread among the orbital planes.
        raan = plane * (360.0 / planes)
        
        # Evenly spaced mean anomaly within the plane, with an offset for relative phasing.
        mean_anomaly = (sat_in_plane * (360.0 / sats_per_plane) +
                        plane * (360.0 / total_sats) * phasing) % 360.0
        
        # rev_num is a placeholder representing the satellite's order in its plane.
        rev_num = sat_in_plane + 1
        
        tle = generate_tle(i + 1, epoch_str, inclination, raan, eccentricity,
                           arg_perigee, mean_anomaly, mean_motion, rev_num)
        tle_list.append(tle)
         
    return fetch_valid_sat_from_tle_list(tle_list)


def generate_walker_constellation_add_planes(sats_per_plane = 100, planes = 2, d_raan = 20, phasing = 1, epoch = datetime(2025, 1, 1),
                                  inclination=50, eccentricity=0.0001, arg_perigee = 0., mean_motion=15.0):
    """
    Generate a Walker Delta constellation by assigning RAAN and mean anomaly for each satellite.
    Satellites are distributed evenly across the planes and within each plane.
    """
    total_sats = sats_per_plane * planes
    epoch_str = format_epoch(epoch)
    tle_list = []
    
    for i in range(total_sats):
        # Determine which orbital plane and the satellite's slot in that plane.
        plane = i % planes         # Plane index: 0 to (planes - 1)
        sat_in_plane = i // planes   # Position within the plane
        
        # RAAN is evenly spread among the orbital planes.
        raan = plane * d_raan
        
        # Evenly spaced mean anomaly within the plane, with an offset for relative phasing.
        mean_anomaly = (sat_in_plane * (360.0 / sats_per_plane) +
                        plane * (360.0 / total_sats) * phasing) % 360.0
        
        # rev_num is a placeholder representing the satellite's order in its plane.
        rev_num = sat_in_plane + 1
        
        tle = generate_tle(i + 1, epoch_str, inclination, raan, eccentricity,
                           arg_perigee, mean_anomaly, mean_motion, rev_num)
        tle_list.append(tle)
         
    return fetch_valid_sat_from_tle_list(tle_list)

def generate_starlink_constellation():
    """
    Generate a Starlink constellation by loading TLE data from a URL.
    """
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    ts, valid_satellites, sat_array = load_url_tle_data(url, reload=True)
    return ts, valid_satellites, sat_array



if __name__ == "__main__":
    # Configuration for the Walker Delta constellation.
    total_sats = 24     # Total number of satellites.
    planes = 6          # Number of orbital planes.
    phasing = 1         # Relative phasing factor.
    
    # Orbital parameters (example values)
    inclination = 55.0       # Orbital inclination in degrees.
    eccentricity = 0.0001     # Nearly circular orbit.
    arg_perigee = 0.0        # Argument of perigee in degrees.
    mean_motion = 15.0       # Mean motion (revolutions per day).
    
    # Define the epoch (using the current UTC time).
    epoch = datetime.utcnow()
    
    # Generate the synthetic TLEs for the constellation.
    constellation = generate_walker_constellation(total_sats, planes, phasing, epoch,
                                                  inclination, eccentricity, arg_perigee, mean_motion)
    
    # Create a Skyfield timescale and get the current time.
    ts = load.timescale()
    t = ts.now()

    # Iterate through the generated TLEs, create EarthSatellite objects,
    # and compute each satellite's geodetic subpoint (latitude, longitude, elevation).
    for satellite in constellation[1]:
        # Create the EarthSatellite object using Skyfield.
        geocentric = satellite.at(t)
        subpoint = wgs84.subpoint(geocentric)
        
        print(f"At time {t.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC, subpoint:")
        print(f"  Latitude:  {subpoint.latitude.degrees:.4f}°")
        print(f"  Longitude: {subpoint.longitude.degrees:.4f}°")
        print(f"  Altitude:  {subpoint.elevation.m:.0f} meters")
        print("")  # Blank line for readability
        
        
    # print(fetch_valid_sat_from_tle_list(constellation))
    print("test datetime, now:",datetime(2025, 1, 1))