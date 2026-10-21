# Delhi Very Last-Mile Delay Intelligence Pipeline

## Project Overview

This project builds a reproducible shipment-delay intelligence
pipeline using synthetic logistics data.

The pipeline cleans shipment records, analyses delivery delays,
predicts delivery hours, classifies potentially delayed shipments,
segments delivery hubs, and generates operational recommendations.

## Data findings

1. **Service type.** Express parcels are delayed most often, at a delay rate of 0.4897
   (119 of 243), compared with 0.2056 for Standard and 0.0857 for Economy.
   Source: `outputs/part_a_delay_by_group.csv`.

2. **Hub tier.** The delay rate rises sharply from Tier 1 (0.0400) to Tier 2 (0.2150)
   to Tier 3 (0.4400).

3. **Distance band.** The delay rate climbs with distance: 0.0219 for 0-200 km,
   0.1368 for 201-400 km, 0.4364 for 401-600 km and 0.8523 for 601+ km.

4. **Strongest correlations with delivery_hours.** The three strongest are
   promised_hours (0.8372), distance_km (0.4013) and n_stops (0.3247),
   ranked by absolute correlation, from `outputs/part_a_correlations.csv`.

5. **Log transform.** Declared value is right-skewed: its mean (1533.05) is far above its
   median (1121.5), while after the log transform they are almost equal
   (7.0277 and 7.0220), so the log makes the distribution roughly symmetric.