# Geographical visualisations

Run the data exploration script from the repository root:

~~~powershell
python src/data_exploration.py --data data/raw --output results --map visualizations/sample_trip_map.html
~~~

The script creates an interactive Folium map with two independently toggled layers:
the observed GPS trajectory and the supplied reference route. The reference geometry
is reconstructed using route arc IDs, arc endpoint node IDs, and node coordinates.

The map is generated from the actual supplied trip data; it is not a manually drawn
or fabricated route. Keep the generated HTML map in this folder when presenting the
dataset-exploration work.
