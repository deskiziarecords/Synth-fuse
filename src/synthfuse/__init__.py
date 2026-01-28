def start_engine():
    """Starts the Cabinet, Librarian, and Physician in sync."""
    print("Synth-Fuse v0.2.0: Cabinet of Alchemists is ONLINE.")
    # Auto-watch the /ingest/ folder
    return cabinet.start_autonomous_loop()
