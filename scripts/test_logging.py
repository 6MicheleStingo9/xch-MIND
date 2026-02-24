import logging
import time
from src.utils.logging import setup_colored_logging

def main():
    setup_colored_logging(level=logging.DEBUG)
    
    # Test different agent loggers
    orchestrator = logging.getLogger("src.agents.orchestrator")
    geo = logging.getLogger("src.agents.workers.geo_analyzer")
    temporal = logging.getLogger("src.agents.workers.temporal_analyzer")
    typological = logging.getLogger("src.agents.workers.type_analyzer")
    path = logging.getLogger("src.agents.workers.path_generator")
    pipeline = logging.getLogger("src.pipeline")
    
    pipeline.info("--- Starting Colored Logging Test ---")
    
    orchestrator.info("Analyzing current state...")
    time.sleep(0.1)
    orchestrator.info("Goal: Identify geographic clusters.")
    time.sleep(0.1)
    orchestrator.info("Dispatching GeoAnalyzerAgent...")
    
    geo.info("Scanning 15 entities for coordinates...")
    time.sleep(0.1)
    geo.debug("Filtering San Silvestro (has coords)...")
    time.sleep(0.1)
    geo.info("Accepted cluster 'Bisceglie' with 4 sites, Radius: 5.2km")
    
    temporal.info("Normalizing 12 period labels...")
    time.sleep(0.1)
    temporal.info("Accepted cluster 'Middle Bronze Age' with 6 sites (Range: -1700 to -1350)")
    
    typological.info("Extracting features from text...")
    time.sleep(0.1)
    typological.info("Accepted typological cluster 'Dolmen a corridoio' with 3 sites")
    
    path.info("Synthesizing thematic paths...")
    time.sleep(0.1)
    path.info("Accepted path 'Apulian Megaliths' (mixed) with 4 stops")
    
    orchestrator.warning("Low confidence on one similarity relation.")
    orchestrator.error("Failed to reach validator agent (mock error).")
    
    pipeline.info("--- Phase Completed ---")

if __name__ == "__main__":
    main()
