import * as maptilersdk from "@maptiler/sdk";
import "@maptiler/sdk/dist/maptiler-sdk.css";
import Box from "@mui/material/Box";
import React, { useEffect, useRef } from "react";
import configData from "./config";
import "./map.css";

// Import your GeoJSON data (you can also fetch it from an API)
const customGeoJSON = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Trois Park",
        "type": "park",
        "rating": 4.2
      },
      "geometry": {
        "type": "Point",
        "coordinates": [87.3218, 23.5502]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "City Center",
        "type": "shopping",
        "rating": 4.0
      },
      "geometry": {
        "type": "Point",
        "coordinates": [87.3089, 23.5386]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Durgapur Barrage",
        "type": "landmark",
        "rating": 4.5
      },
      "geometry": {
        "type": "Point",
        "coordinates": [87.3486, 23.5167]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Nehru Park",
        "type": "park",
        "rating": 3.9
      },
      "geometry": {
        "type": "Point",
        "coordinates": [87.3025, 23.5278]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Durgapur Steel Plant",
        "type": "industrial",
        "rating": 3.7
      },
      "geometry": {
        "type": "Point",
        "coordinates": [87.2769, 23.5153]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "B Zone Park",
        "type": "park",
        "rating": 4.1
      },
      "geometry": {
        "type": "Point",
        "coordinates": [87.2864, 23.5333]
      }
    }
  ]
};

export default function Map() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const center = { lng: -157.9253, lat: 21.4732 };
  const zoom = 9.79;
  maptilersdk.config.apiKey = configData.MAPTILER_API_KEY;

  useEffect(() => {
    if (map.current) return;

    map.current = new maptilersdk.Map({
      container: mapContainer.current,
      style: maptilersdk.MapStyle.DATAVIZ.LIGHT,
      center: [center.lng, center.lat],
      zoom: zoom,
      hash: true,
    });

    map.current.on("load", () => {
      // Add your GeoJSON source
      map.current.addSource('custom-geojson', {
        type: 'geojson',
        data: customGeoJSON // or you could fetch it: 'https://yourdomain.com/data.geojson'
      });

      // Add a layer for your GeoJSON data
      map.current.addLayer({
        id: 'custom-data-layer',
        type: 'circle', // or 'fill', 'line', 'symbol' depending on your data
        source: 'custom-geojson',
        paint: {
          'circle-color': [
            'interpolate',
            ['linear'],
            ['get', 'minimum_nights'], // use your property name
            0, '#2DC4B2',
            10, '#3BB3C3',
            20, '#669EC4',
            30, '#8B88B6',
            40, '#A2719B'
          ],
          'circle-opacity': 0.8,
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['zoom'],
            5, 2,
            15, 8
          ]
        }
      });

      // Optional: Add clustering if you have many points
      /*
      map.current.addSource('custom-geojson-clustered', {
        type: 'geojson',
        data: customGeoJSON,
        cluster: true,
        clusterMaxZoom: 14,
        clusterRadius: 50
      });
      
      // Add cluster layers...
      */
    });
  }, [center.lng, center.lat, zoom]);

  return (
    <Box sx={{ display: "flex" }}>
      <div className="container">
        <div ref={mapContainer} id="map" className="map" />
      </div>
    </Box>
  );
}