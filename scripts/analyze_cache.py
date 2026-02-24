import json
import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

def analyze_cache():
    # Set paths relative to project root
    base_dir = Path(__file__).parent.parent
    entities_dir = base_dir / "entities"
    cache_path = entities_dir / "sparql_cache.json"
    output_report_path = base_dir / "output" / "cache_analysis_report.json"
    
    if not cache_path.exists():
        print(f"Error: Cache not found at {cache_path}")
        return

    with open(cache_path, "r") as f:
        cache = json.load(f)
    
    # Extract all known geometries from cache
    # The cache structure is usually { "some_hash": { "geometries": { ... }, "wikidata_coordinates": { ... } } }
    # Let's flatten all geometries from all hash keys
    cached_geos = {}
    wikidata_coords = {}
    
    for key in cache:
        if isinstance(cache[key], dict):
            if "geometries" in cache[key]:
                cached_geos.update(cache[key]["geometries"])
            if "wikidata_coordinates" in cache[key]:
                wikidata_coords.update(cache[key]["wikidata_coordinates"])

    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'clv': 'https://w3id.org/italia/onto/CLV/',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'a-loc': 'https://w3id.org/arco/ontology/location/'
    }

    results = []
    xml_files = list(entities_dir.glob("*.xml"))
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Find the main entity URI
            desc = root.find(".//rdf:Description[@rdf:about]", ns)
            if desc is None:
                continue
            
            entity_uri = desc.get(f"{{{ns['rdf']}}}about")
            entity_name = xml_file.stem
            
            # Find municipality for fallback
            coverage = root.find(".//dc:coverage", ns)
            municipality = None
            if coverage is not None and coverage.text:
                m = re.match(r"([^(]+)", coverage.text)
                if m:
                    municipality = m.group(1).strip().lower()

            # Find all geometry URIs
            geo_els = root.findall(".//clv:hasGeometry[@rdf:resource]", ns)
            geo_uris = [el.get(f"{{{ns['rdf']}}}resource") for el in geo_els]
            
            has_coords = False
            geo_source = None
            
            # Check specific geometries
            for g_uri in geo_uris:
                if g_uri in cached_geos:
                    has_coords = True
                    geo_source = "ArCo (Geometry URI)"
                    break
            
            # Check municipality fallback
            if not has_coords and municipality and municipality in wikidata_coords:
                has_coords = True
                geo_source = f"Wikidata (Municipality: {municipality})"
            
            results.append({
                "file": xml_file.name,
                "name": entity_name,
                "uri": entity_uri,
                "has_coordinates": has_coords,
                "source": geo_source,
                "municipality": municipality
            })
            
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")

    # Report
    missing = [r for r in results if not r["has_coordinates"]]
    found = [r for r in results if r["has_coordinates"]]
    
    print(f"Total entities analyzed: {len(results)}")
    print(f"Entities WITH coordinates: {len(found)}")
    print(f"Entities MISSING coordinates: {len(missing)}")
    
    if missing:
        print("\nMissing Coordinates Detail:")
        for m in missing:
            print(f"  - {m['file']} ({m['municipality'] or 'unknown municipality'})")
    
    # Save a detailed report
    report = {
        "stats": {
            "total": len(results),
            "with_coords": len(found),
            "missing_coords": len(missing)
        },
        "missing": missing,
        "found": found
    }
    
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to {output_report_path}")

if __name__ == "__main__":
    analyze_cache()
