import math
from datetime import datetime
from skyfield.api import load, EarthSatellite, wgs84
from sgp4.api import SatrecArray
import numpy as np


def _starlink_orbital_features(valid_satellites):
    inclinations_deg = np.array(
        [sat.model.inclo * 180.0 / math.pi for sat in valid_satellites],
        dtype=np.float64,
    )
    mean_motion_rev_per_day = np.array(
        [sat.model.no_kozai * 1440.0 / (2.0 * math.pi) for sat in valid_satellites],
        dtype=np.float64,
    )
    raan_deg = np.mod(
        np.array([sat.model.nodeo * 180.0 / math.pi for sat in valid_satellites], dtype=np.float64),
        360.0,
    )
    return inclinations_deg, mean_motion_rev_per_day, raan_deg


def _select_circular_contiguous_block(raan_deg, n_sat):
    if n_sat > raan_deg.size:
        raise ValueError("Requested block size exceeds the number of satellites in the shell cluster.")

    order = np.argsort(raan_deg)
    sorted_raan = raan_deg[order]
    extended_raan = np.concatenate((sorted_raan, sorted_raan + 360.0))

    best_start = 0
    best_span = np.inf
    for start in range(sorted_raan.size):
        end = start + n_sat - 1
        span = extended_raan[end] - extended_raan[start]
        if span < best_span:
            best_span = span
            best_start = start

    best_indices = order[np.mod(np.arange(best_start, best_start + n_sat), sorted_raan.size)]
    return best_indices, best_span


def select_starlink_shell_block_indices(
    valid_satellites,
    n_sat=1000,
    target_inclination_deg=53.2,
    inclination_window_deg=0.3,
    inclination_round_deg=0.02,
    mean_motion_round_rev_per_day=0.003,
):
    inclinations_deg, mean_motion_rev_per_day, raan_deg = _starlink_orbital_features(valid_satellites)

    rounded_inclinations = np.round(inclinations_deg / inclination_round_deg) * inclination_round_deg
    rounded_mean_motion = (
        np.round(mean_motion_rev_per_day / mean_motion_round_rev_per_day)
        * mean_motion_round_rev_per_day
    )

    target_shell_mask = np.abs(rounded_inclinations - target_inclination_deg) <= inclination_window_deg
    if not np.any(target_shell_mask):
        raise ValueError("No Starlink shell cluster was found near the requested inclination.")

    shell_keys, shell_counts = np.unique(
        np.column_stack((rounded_inclinations[target_shell_mask], rounded_mean_motion[target_shell_mask])),
        axis=0,
        return_counts=True,
    )
    shell_key = shell_keys[np.argmax(shell_counts)]
    shell_mask = (
        np.isclose(rounded_inclinations, shell_key[0])
        & np.isclose(rounded_mean_motion, shell_key[1])
    )

    shell_indices = np.flatnonzero(shell_mask)
    if shell_indices.size < n_sat:
        raise ValueError(
            "The selected Starlink shell cluster does not contain enough satellites for the requested block."
        )

    block_local_indices, block_span_deg = _select_circular_contiguous_block(raan_deg[shell_indices], n_sat)
    selected_indices = shell_indices[block_local_indices]
    selected_indices = selected_indices[np.argsort(raan_deg[selected_indices])]

    metadata = {
        "shell_inclination_deg": float(shell_key[0]),
        "shell_mean_motion_rev_per_day": float(shell_key[1]),
        "shell_population": int(shell_indices.size),
        "selected_population": int(selected_indices.size),
        "raan_span_deg": float(block_span_deg),
    }
    return selected_indices, metadata


def generate_tle_starlink_shell_block_constellation(
    n_sat=1000,
    starlink_tle_path=None,
    target_inclination_deg=53.2,
    inclination_window_deg=0.3,
    inclination_round_deg=0.02,
    mean_motion_round_rev_per_day=0.003,
):
    ts, valid_satellites_starlink, _ = load_url_tle_data(starlink_tle_path, reload=True)
    selected_indices, metadata = select_starlink_shell_block_indices(
        valid_satellites_starlink,
        n_sat=n_sat,
        target_inclination_deg=target_inclination_deg,
        inclination_window_deg=inclination_window_deg,
        inclination_round_deg=inclination_round_deg,
        mean_motion_round_rev_per_day=mean_motion_round_rev_per_day,
    )

    valid_satellites = [valid_satellites_starlink[idx] for idx in selected_indices]
    models = [sat.model for sat in valid_satellites]
    sat_array = SatrecArray(models)
    return ts, valid_satellites, sat_array, metadata

def generate_tle_partly_regular_constellation1000(n_sat=1000, ratio=0.5, starlink_tle_path=None, seed=0):
    N_SAT = n_sat
    rng = np.random.default_rng(seed=seed)
    ts, valid_satellites_starlink, sat_array = load_url_tle_data(starlink_tle_path, reload=True)
    idx = rng.choice(np.arange(len(valid_satellites_starlink)), size=N_SAT, replace=False)
    valid_satellites_starlink = [valid_satellites_starlink[x] for x in idx]
    models = [sat.model for sat in valid_satellites_starlink]
    sat_array = SatrecArray(models)
    
    N_REGULAR = int(N_SAT * ratio)
    N_STARLINK = N_SAT - N_REGULAR
    
    idx = rng.choice(np.arange(len(valid_satellites_starlink)), size=N_STARLINK, replace=False)
    valid_satellites_starlink = [valid_satellites_starlink[x] for x in idx]
    
    N_REGULAR_ORBITS = int(math.sqrt(N_SAT / 2))
    N_REGULAR_SAT_PER_ORBIT = int(N_SAT / N_REGULAR_ORBITS)
    while N_REGULAR_SAT_PER_ORBIT * N_REGULAR_ORBITS < N_SAT:
        N_REGULAR_SAT_PER_ORBIT += 1
    ts, valid_satellites_regular, sat_array = generate_walker_constellation(sats_per_plane=N_REGULAR_SAT_PER_ORBIT, planes=N_REGULAR_ORBITS)
    print("Generated %d regular satellites in %d orbits with %d satellites per orbit." % (len(valid_satellites_regular), N_REGULAR_ORBITS, N_REGULAR_SAT_PER_ORBIT))
    if N_REGULAR == 0:
        valid_satellites_regular = []
    else:   
        idx = rng.choice(np.arange(len(valid_satellites_regular)), size=N_REGULAR, replace=False)
        valid_satellites_regular = [valid_satellites_regular[x] for x in idx]
    
    valid_satellites = valid_satellites_regular + valid_satellites_starlink
    models = [sat.model for sat in valid_satellites]
    sat_array = SatrecArray(models)    

    return ts, valid_satellites, sat_array


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


def load_tle_valid_at_times(path, times):
    """Load a historical catalogue, retaining records valid at every supplied epoch.

    Unlike load_url_tle_data, this opt-in loader never filters at wall-clock now.
    SGP4 uses UTC Julian dates, matching Skyfield's satellite implementation.
    """
    from sgp4.api import jday
    ts = load.timescale(builtin=True)
    satellites = load.tle_file(str(path), reload=False)
    dates = [jday(*t.utc) for t in times]
    jd, fraction = np.asarray(dates, dtype=float).T
    errors, positions, velocities = SatrecArray([s.model for s in satellites]).sgp4(
        np.ascontiguousarray(jd), np.ascontiguousarray(fraction))
    valid = ((errors == 0).all(axis=1)
             & np.isfinite(positions).all(axis=(1, 2))
             & np.isfinite(velocities).all(axis=(1, 2)))
    kept = [s for s, ok in zip(satellites, valid) if ok]
    ids = [s.model.satnum for s in kept]
    if len(ids) != len(set(ids)):
        raise ValueError('Catalogue contains duplicate satellite IDs')
    return ts, kept, {'catalogue_population': len(satellites),
                      'valid_population': len(kept),
                      'excluded_ids': [s.model.satnum for s, ok in zip(satellites, valid) if not ok]}


def select_shifted_shell_block(satellites, inclination, mean_motion, n_sat, selection):
    """Select one of three predetermined circular RAAN blocks in a TLE cluster."""
    if selection not in (0, 1, 2):
        raise ValueError('selection must be 0, 1, or 2')
    inc, motion, raan = _starlink_orbital_features(satellites)
    members = np.flatnonzero(np.isclose(np.round(inc / .02) * .02, inclination)
                            & np.isclose(np.round(motion / .003) * .003, mean_motion))
    if len(members) < n_sat:
        raise ValueError(f'Cluster {(inclination, mean_motion)} has {len(members)}, needs {n_sat}')
    order = np.argsort(raan[members])
    first, _ = _select_circular_contiguous_block(raan[members], n_sat)
    minimum_start = int(np.flatnonzero(order == first[0])[0])
    shift = selection * len(members) // 3
    start = (minimum_start + shift) % len(members)
    selected = members[order[(start + np.arange(n_sat)) % len(members)]]
    # Keep the historical selector's final RAAN ordering for reproducibility.
    selected = selected[np.argsort(raan[selected])]
    circular = raan[members[order[(start + np.arange(n_sat)) % len(members)]]]
    return selected, {'inclination_deg': inclination, 'mean_motion_rev_per_day': mean_motion,
                      'cluster_population': len(members), 'selected_population': n_sat,
                      'minimum_start_index': minimum_start, 'start_index': start,
                      'shift': shift, 'raan_span_deg': float((circular[-1] - circular[0]) % 360)}
