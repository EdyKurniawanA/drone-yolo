from logger import CSVLogger


def main():
    logger = CSVLogger("phase3_test_log.csv")

    # Simulate writing some rows
    logger.log(
        frame_id=1,
        class_name="person",
        class_count=3,
        lat=-7.2453,
        lon=112.7378,
        cpu=45,
        ram=63,
        gpu=72,
    )

    logger.log(
        frame_id=2,
        class_name="car",
        class_count=1,
        lat=-7.2455,
        lon=112.7380,
        cpu=42,
        ram=61,
        gpu=70,
    )

    logger.log(
        frame_id=3,
        class_name="dog",
        class_count=2,
        lat=-7.2460,
        lon=112.7382,
        cpu=50,
        ram=65,
        gpu=75,
    )

    print("✅ Logging test complete. Check 'phase3_test_log.csv'.")


if __name__ == "__main__":
    main()
