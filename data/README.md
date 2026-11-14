# Case-study data (hand-downloaded, weeks 1-2)

Per the build order, the integration stages are developed on hand-downloaded open data before any agent does acquisition. Put files in `data/raw/`; this folder's contents are git-ignored.

For each file, record in `data/raw/SOURCES.md`: URL, download date, licence shown on the page, row count, and whether it duplicates another file.

## Case 1: Delhi house prices
| File | Where | Role | Notes |
|---|---|---|---|
| `delhi_magicbricks.csv` | Kaggle, search "MagicBricks Delhi house price" (1,259 rows × 11 cols: Area, BHK, Bathroom, Furnishing, Locality, Parking, Price, Status, Transaction, Type, Per_Sqft) | entity | `Per_Sqft` = Price / Area: the known leakage test (E10). Check the licence on the Kaggle page. |
| a 2nd Delhi/NCR sale-listing dataset | Kaggle, search "Delhi NCR real estate" | entity | Check whether it is a re-upload of the same scrape before treating it as independent. |
| `delhi_metro_stations.geojson` | Overpass Turbo, query `railway=station` + `station=subway` in Delhi | context (join on coordinates) | ODbL licence, attribute OpenStreetMap. |
| `delhi_circle_rates` (PDF/HTML) | revenue.delhi.gov.in, colony categories A-H | context (join on locality) | Government source (authority 1.0). Same colony name can sit in different categories in different districts. |
| locality coordinates | Nominatim/OSM geocoding of locality names | context | Respect the Nominatim usage policy (1 request per second, cache results). |

## Case 2: Laptop prices
| File | Where | Role | Notes |
|---|---|---|---|
| `laptop_price.csv` | Kaggle "Laptop Price" (Muhammet Varlı), ~1,303 rows, prices in EUR | entity | About 2018 data: exercises currency normalisation and recency weighting. |

## Out of scope
Live scraping of 99acres, MagicBricks or Housing.com (decision D4).
