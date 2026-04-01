import hashlib
import pickle


def load_samples(config, zones_dict):
    """
    Load predefined samples from a JSON file and add them to the zones.
    """
    import os

    samples_path = config.sampling.samples_path
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    all_samples = []
    with open(samples_path, "rb") as f:
        samples_data = pickle.load(f)
        # hashed_tables = [hashlib.md5(table.encode()).hexdigest() for table in tables]
        for table, samples in samples_data[1][config.labeling.labeling_budget].items():
            for sample in samples:
                all_samples.append(
                    (hashlib.md5(sample[0].encode()).hexdigest(), sample[2], sample[1])
                )
        print("")

    all_cols = []
    for sample in all_samples:
        if (sample[0], sample[1]) not in all_cols:
            all_cols.append((sample[0], sample[1]))

    cells_to_zones = {}
    for zone in zones_dict.values():
        for cell_id, cell in zone.cells.items():
            cells_to_zones[cell_id] = zone.name

    for sample in all_samples:
        zone = cells_to_zones.get(sample, None)
        if zone is not None:
            zones_dict[zone].samples[sample] = zones_dict[zone].cells.get(sample, None)

    zone_samples = 0
    for zone in zones_dict.values():
        zone_samples += len(zone.samples)

    print("")
