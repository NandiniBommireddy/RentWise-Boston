import type { RagResponse } from '../types'

/**
 * Mock of the RentWise backend. Matches the intended contract:
 *   POST /api/ask { query } -> RagResponse
 * Swap the body of askRentWise() for a real fetch() when the backend exists.
 *
 * All addresses/coordinates below are real rows from the RentSmart download
 * (data/downloads/), so the map behaves like it will in production.
 */

const LATENCY_MS = () => 700 + Math.random() * 600

const scenarios: { pattern: RegExp; response: RagResponse }[] = [
  {
    pattern: /rodent|rat\b|rats|mice|mouse|pest/i,
    response: {
      answer:
        'Rodent activity is one of the most common sanitation issues in the RentSmart data. ' +
        'In the past week alone, rodent-related reports were filed across Brighton, Allston, ' +
        'Roxbury, East Boston, Jamaica Plain, and Mission Hill. The most recent report was at ' +
        '44 Portsmouth St in Brighton (a 2-family built in 1900), and Mission Hill logged two ' +
        'reports on Bickford Ave within minutes of each other, which often indicates a shared ' +
        'infestation across adjacent buildings. Pest issues appear in the data both as ' +
        'Sanitation Requests ("Rodent Activity") and Housing Complaints ("Pest Infestation - Residential").',
      locations: [
        { address: '44 Portsmouth St, 02135', neighborhood: 'Brighton', latitude: 42.35934, longitude: -71.14419, label: 'Rodent Activity', details: 'Sanitation Request · Aug 21, 2026 · Residential 2-family (1900)' },
        { address: '15 Chester St, 02134', neighborhood: 'Allston', latitude: 42.35124, longitude: -71.12853, label: 'Rodent Activity', details: 'Sanitation Request · Aug 21, 2026 · Residential 4+ family (1920)' },
        { address: '19 E Springfield St, 02118', neighborhood: 'Roxbury', latitude: 42.33634, longitude: -71.07582, label: 'Rodent Activity', details: 'Sanitation Request · Aug 21, 2026 · Residential 4+ family (1899)' },
        { address: '46 Geneva St, 02128', neighborhood: 'East Boston', latitude: 42.369512, longitude: -71.032488, label: 'Rodent Activity', details: 'Sanitation Request · Aug 20, 2026 · Condominium (2018)' },
        { address: '186 Heath St, 02130', neighborhood: 'Jamaica Plain', latitude: 42.325973, longitude: -71.105069, label: 'Rodent Activity', details: 'Sanitation Request · Aug 20, 2026 · Mixed Use (1926)' },
        { address: '10 Bickford Ave, 02120', neighborhood: 'Mission Hill', latitude: 42.32686, longitude: -71.10258, label: 'Rodent Activity', details: 'Sanitation Request · Aug 20, 2026 · Residential 4+ family (1905)' },
        { address: '6 Bickford Ave, 02120', neighborhood: 'Mission Hill', latitude: 42.32668, longitude: -71.10252, label: 'Rodent Activity', details: 'Sanitation Request · Aug 20, 2026 · Residential 3-family (1900)' },
      ],
      sources: ['Sanitation Requests: Rodent Activity', 'Housing Complaints: Pest Infestation - Residential'],
    },
  },
  {
    pattern: /heat|heating|cold|temperature/i,
    response: {
      answer:
        'Heating complaints are recorded as Housing Complaints with the description ' +
        '"Heat - Excessive, Insufficient". Recent examples include 1 Rosa St in Hyde Park ' +
        '(a 7+ unit building owned by GBM Portfolio Owner LLC), the Old Colony Phase Three ' +
        'condominium at 20 Rev Richard A Burke St in South Boston, and 19 Pleasant St in ' +
        'Dorchester. Heat complaints cluster in winter months, but off-season filings like ' +
        'these July/August reports usually indicate excessive heat or long-running disputes.',
      locations: [
        { address: '1 Rosa St, 02136', neighborhood: 'Hyde Park', latitude: 42.26162, longitude: -71.11198, label: 'Heat - Excessive, Insufficient', details: 'Housing Complaint · Aug 7, 2026 · Residential 7+ units (1965)' },
        { address: '20 Rev Richard A Burke St, 02127', neighborhood: 'South Boston', latitude: 42.331899, longitude: -71.051741, label: 'Heat - Excessive, Insufficient', details: 'Housing Complaint · Aug 3, 2026 · Condominium' },
        { address: '16 Holmfield Ave, 02136', neighborhood: 'Mattapan', latitude: 42.26341, longitude: -71.10027, label: 'Heat - Excessive, Insufficient', details: 'Housing Complaint · Jul 8, 2026 · Residential 1-family (1910)' },
        { address: '19 Pleasant St 2, 02125', neighborhood: 'Dorchester', latitude: 42.31742, longitude: -71.05946, label: 'Heat - Excessive, Insufficient', details: 'Housing Complaint · Jul 7, 2026 · Residential 3-family (1905)' },
      ],
      sources: ['Housing Complaints: Heat - Excessive, Insufficient'],
    },
  },
  {
    pattern: /unsafe|dangerous|structur|collapse|hazard/i,
    response: {
      answer:
        'The dataset tracks unsafe conditions in two categories: Housing Complaints ' +
        '("Unsafe Dangerous Conditions") and Building Violations ("Unsafe Structures"). ' +
        'Recent reports include 74 Marine Rd in South Boston, a 3-family owned by Real ' +
        'Estate Boston LLC that was remodeled as recently as 2023, and an Unsafe Structures ' +
        'building violation at 12 Neponset Ave in Hyde Park. Older housing stock dominates ' +
        'these reports — every recent example was built before 1920.',
      locations: [
        { address: '74 Marine Rd, 02127', neighborhood: 'South Boston', latitude: 42.33079, longitude: -71.03632, label: 'Unsafe Dangerous Conditions', details: 'Housing Complaint · Aug 15, 2026 · Residential 3-family (1905, remod. 2023)' },
        { address: '12 Neponset Ave, 02136', neighborhood: 'Hyde Park', latitude: 42.25035, longitude: -71.12038, label: 'Unsafe Structures', details: 'Building Violation · Aug 12, 2026 · Residential 2-family (1915)' },
        { address: '117-119 Fulton St, 02109', neighborhood: 'Boston', latitude: 42.36249, longitude: -71.05282, label: 'Unsafe Dangerous Conditions', details: 'Housing Complaint · Aug 11, 2026 · Condominium (1899)' },
        { address: '103 Colberg Ave, 02131', neighborhood: 'Roslindale', latitude: 42.28361, longitude: -71.14367, label: 'Unsafe Dangerous Conditions', details: 'Housing Complaint · Aug 10, 2026 · Residential 1-family (1895)' },
      ],
      sources: ['Housing Complaints: Unsafe Dangerous Conditions', 'Building Violations: Unsafe Structures'],
    },
  },
  {
    pattern: /dorchester/i,
    response: {
      answer:
        'Dorchester is one of the most active neighborhoods in the RentSmart data. Recent ' +
        'reports span abandoned vehicles on Neponset Ave and Glendale St, rodent activity at ' +
        '4 Hartford Ct, and a mice infestation complaint at 67 Whitten St (a 3-family built ' +
        'in 1920). Sanitation Requests and Housing Complaints make up the bulk of recent ' +
        'Dorchester records, and most affected properties are 2- and 3-family homes built ' +
        'before 1925.',
      locations: [
        { address: '289 Neponset Ave, 02122', neighborhood: 'Dorchester', latitude: 42.28902, longitude: -71.04783, label: 'Abandoned Vehicles', details: 'Sanitation Request · Aug 20, 2026 · Residential 2-family (1900)' },
        { address: '4 Hartford Ct, 02125', neighborhood: 'Dorchester', latitude: 42.31424, longitude: -71.07258, label: 'Rodent Activity', details: 'Sanitation Request · Aug 20, 2026 · Residential 2-family (1920)' },
        { address: '67 Whitten St, 02122', neighborhood: 'Dorchester', latitude: 42.2936, longitude: -71.05781, label: 'Mice Infestation - Residential', details: 'Housing Complaint · Aug 20, 2026 · Residential 3-family (1920)' },
        { address: '12 Glendale St, 02125', neighborhood: 'Dorchester', latitude: 42.31318, longitude: -71.06564, label: 'Abandoned Vehicles', details: 'Sanitation Request · Aug 20, 2026 · Residential 2-family (1910)' },
      ],
      sources: ['RentSmart records where neighborhood = Dorchester'],
    },
  },
]

const fallback: RagResponse = {
  answer:
    'The RentSmart dataset contains 389,569 records of housing-related activity across ' +
    'Boston. The largest categories are Enforcement Violations (280,197), Housing ' +
    'Complaints (48,864), Sanitation Requests (38,781), Housing Violations (17,699), and ' +
    'Building Violations (3,109). Each record includes the address, neighborhood, property ' +
    'owner, year built, and property type. Try asking about a specific issue (rodents, ' +
    'heat, unsafe conditions) or a neighborhood (e.g. Dorchester) to see mapped results.',
  locations: [],
  sources: ['RentSmart dataset summary'],
}

export async function askRentWise(query: string): Promise<RagResponse> {
  // Real implementation:
  //   const res = await fetch('/api/ask', { method: 'POST', headers: {...}, body: JSON.stringify({ query }) })
  //   return await res.json()
  await new Promise((resolve) => setTimeout(resolve, LATENCY_MS()))
  const match = scenarios.find((s) => s.pattern.test(query))
  return structuredClone(match ? match.response : fallback)
}
