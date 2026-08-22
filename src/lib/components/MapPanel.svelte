<script lang="ts">
  import mapboxgl from 'mapbox-gl'
  import 'mapbox-gl/dist/mapbox-gl.css'
  import type { RagLocation } from '../types'

  let { locations = [] }: { locations?: RagLocation[] } = $props()

  const token: string = import.meta.env.VITE_MAPBOX_TOKEN ?? ''
  const BOSTON_CENTER: [number, number] = [-71.0589, 42.3601]

  let container = $state<HTMLDivElement>()
  let map: mapboxgl.Map | null = null
  let markers: mapboxgl.Marker[] = []

  $effect(() => {
    if (!token || !container) return
    mapboxgl.accessToken = token
    map = new mapboxgl.Map({
      container,
      style: 'mapbox://styles/mapbox/light-v11',
      center: BOSTON_CENTER,
      zoom: 11.3,
    })
    map.addControl(new mapboxgl.NavigationControl(), 'top-right')
    return () => {
      map?.remove()
      map = null
    }
  })

  // Sync markers whenever the locations prop changes.
  $effect(() => {
    const locs = locations
    if (!map) return

    markers.forEach((m) => m.remove())
    markers = locs.map((loc) => {
      const popup = new mapboxgl.Popup({ offset: 24, maxWidth: '280px' }).setHTML(
        `<strong>${escapeHtml(loc.label)}</strong><br>${escapeHtml(loc.address)}<br>` +
          `<span class="popup-muted">${escapeHtml(loc.neighborhood)}${loc.details ? ' · ' + escapeHtml(loc.details) : ''}</span>`
      )
      return new mapboxgl.Marker({ color: '#c2410c' })
        .setLngLat([loc.longitude, loc.latitude])
        .setPopup(popup)
        .addTo(map!)
    })

    if (locs.length === 1) {
      map.flyTo({ center: [locs[0].longitude, locs[0].latitude], zoom: 15.5, duration: 1200 })
    } else if (locs.length > 1) {
      const bounds = new mapboxgl.LngLatBounds()
      locs.forEach((l) => bounds.extend([l.longitude, l.latitude]))
      map.fitBounds(bounds, { padding: 70, maxZoom: 15, duration: 1200 })
    } else {
      map.flyTo({ center: BOSTON_CENTER, zoom: 11.3, duration: 1200 })
    }
  })

  function escapeHtml(s: string): string {
    return s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`)
  }
</script>

{#if token}
  <div class="map" bind:this={container}></div>
{:else}
  <div class="map token-missing">
    <div>
      <h3>Mapbox token needed</h3>
      <p>
        Copy <code>.env.example</code> to <code>.env</code>, set
        <code>VITE_MAPBOX_TOKEN</code> to your Mapbox access token, and restart the dev
        server to see the map.
      </p>
    </div>
  </div>
{/if}

<style>
  .map {
    width: 100%;
    height: 100%;
    min-height: 0;
    border-radius: 1rem;
    overflow: hidden;
  }
  .token-missing {
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--surface);
    border: 1.5px dashed var(--border);
    padding: 2rem;
    text-align: center;
  }
  .token-missing h3 {
    margin: 0 0 0.5rem;
  }
  .token-missing p {
    margin: 0;
    color: var(--text-muted);
    line-height: 1.6;
    max-width: 34ch;
  }
  code {
    background: var(--accent-soft);
    padding: 0.1em 0.35em;
    border-radius: 0.3em;
    font-size: 0.9em;
  }
  :global(.mapboxgl-popup-content) {
    font-family: inherit;
    line-height: 1.5;
    padding: 0.75rem 1rem;
    border-radius: 0.6rem;
  }
  :global(.popup-muted) {
    color: #6b7280;
    font-size: 0.85em;
  }
</style>
