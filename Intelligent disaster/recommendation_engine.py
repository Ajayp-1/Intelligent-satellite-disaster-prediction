from typing import List, Dict, Any

SEVERITY = {
    "low": (0.0, 0.4),
    "medium": (0.4, 0.7),
    "high": (0.7, 0.85),
    "critical": (0.85, 1.01),
}

RIVER_BASIN_MAP = {
    "Chennai": ["Adyar River Basin", "Cooum River Basin", "Buckingham Canal"],
    "Cuddalore": ["Ponnaiyar River Basin", "Vellar River Basin", "Gadilam River"],
    "Thiruvallur": ["Cooum River upper basin", "Arani River Basin", "Kosasthalaiyar"],
    "Kanchipuram": ["Palar River Basin", "Cheyyar River"],
    "Nagapattinam": ["Vennar River Delta", "Cauvery Delta tailwaters", "Coleroon River"],
    "Thanjavur": ["Cauvery River Delta", "Vennar River", "Grand Anicut channels"],
    "Tiruchirappalli": ["Cauvery River Basin", "Kollidam River"],
    "Tirunelveli": ["Thamirabarani River Basin", "Manimuthar River"],
    "Madurai": ["Vaigai River Basin", "Gundar River"],
    "Coimbatore": ["Bhavani River Basin", "Noyyal River"],
    "Salem": ["Thirumanimutharu River", "Vasishta Nadi"],
    "Erode": ["Cauvery River", "Bhavani River"],
    "Vellore": ["Palar River Basin", "Cheyyar River upper basin"],
    "Kanyakumari": ["Thamirabarani headwaters", "Pazhaiyar River", "Kodayar River"],
    "Nilgiris": ["Moyar River Basin", "Bhavani headwaters", "Coonoor River"],
}

COASTAL_CYCLONE_DISTRICTS = {
    "Nagapattinam", "Cuddalore", "Chennai", "Kanyakumari", "Thiruvallur", "Thanjavur",
}

NILGIRI_HILL_ROADS = [
    "Mettupalayam-Ooty NH 181 (Kotagiri Ghat Road)",
    "Coonoor-Ooty SH 15",
    "Gudalur-Ooty NH 67 (Mysore Road)",
    "Kotagiri-Kodanadu SH 15",
    "Coonoor-Mettupalayam Ghat section",
]

FLOOD_SLUM_AREAS = {
    "Chennai": [
        "Velachery-Taramani low-lying residential belts",
        "Vyasarpadi, Washermanpet, Tondiarpet (north Chennai slums)",
        "Adyar / Guindy floodplains",
        "Pallikaranai marsh peripheries",
        "Koyambedu-Annanagar drains",
    ],
    "Cuddalore": [
        "Cuddalore OT (Old Town) slum clusters",
        "Pudupettai & Parangipettai riverfront settlements",
        "Neyveli surrounding lowlands",
        "Chidambaram eastern wards",
    ],
    "Thiruvallur": [
        "Tiruvottiyur coastal slums",
        "Minjur-Ponneri delta hamlets",
        "Ponneri-Sriperumbudur belt",
        "Redhills lake perimeter villages",
    ],
    "Nagapattinam": [
        "Nagapattinam town tidal-zone hamlets",
        "Velankanni-Thiruthuraipoondi delta villages",
        "Tharangambadi coastal blocks",
    ],
    "Kanchipuram": [
        "Sriperumbudur low-lying industrial townships",
        "Chengalpattu-Kanchipuram road flood pockets",
    ],
}

TANGEDCO_FEEDERS = {
    "Nagapattinam": "TANGEDCO Nagapattinam Circle – Mayiladuthurai, Karaikal feeders",
    "Cuddalore": "TANGEDCO Cuddalore Circle – Neyveli, Chidambaram HT lines",
    "Chennai": "TANGEDCO Chennai Metro – Tondiarpet, Kodungaiyur substations",
    "Thiruvallur": "TANGEDCO Tiruvallur Circle – Minjur, Ponneri feeders",
    "Kanyakumari": "TANGEDCO Kanniyakumari Circle – Nagercoil, Marthandam lines",
}


def _risk_label(score: float) -> str:
    for label, (lo, hi) in SEVERITY.items():
        if lo <= score < hi:
            return label
    return "critical"


def _severity_action_prefix(severity: str) -> str:
    return {
        "low": "Monitor closely. Activate standby teams.",
        "medium": "Elevate to Block-level EOC. Pre-position resources.",
        "high": "DEFCON-2. Activate District EOC. Begin proactive evacuations.",
        "critical": "DEFCON-1. Immediate evacuation. Call in Army / NDRF columns.",
    }.get(severity, "Monitor and assess.")


def generate_tn_recommendations(district: str, disaster_type: str, risk_score: float) -> Dict[str, Any]:
    district = str(district).strip()
    disaster_type = str(disaster_type).strip()
    risk_score = max(0.0, min(1.0, float(risk_score)))
    severity = _risk_label(risk_score)
    prefix = _severity_action_prefix(severity)

    actions: List[str] = [prefix]
    agencies: List[str] = ["TN State Disaster Management Authority (SDMA)", "District Emergency Operations Centre (DEOC)"]
    helplines = {
        "TN Emergency Services": "108",
        "TN Fire & Rescue Services": "101",
        "TN Police Control Room": "100",
        "TNGDMA Helpline": "1070",
        "NDRF Helpline": "9711077372",
    }

    if disaster_type == "Safe":
        return {
            "severity": "safe",
            "alert_title": f"✅ {district} — Normal Conditions",
            "summary": "All environmental indicators within safe thresholds. Routine monitoring continues.",
            "key_actions": [
                "Maintain routine daily monitoring of rainfall & river gauges.",
                "Ensure SDMA-DMA communication link health checks.",
                "Replenish shelter stock inventories (buses, motor pumps, first-aid kits) on schedule.",
            ],
            "agencies": ["Revenue & Disaster Management Dept. (routine monitoring)"],
            "helplines": helplines,
            "infra": [],
        }

    if disaster_type == "Flood":
        rivers = RIVER_BASIN_MAP.get(district, ["Major surface water bodies and storm-water drains"])
        slums = FLOOD_SLUM_AREAS.get(district, [f"Low-lying settlements across {district}"])
        actions += [
            f"Activate real-time telemetric monitoring of {', '.join(rivers)}.",
            f"Issue flash-flood SMS/IVRS alerts (TN-AWS + RSO) to 5-km buffers of {', '.join(rivers)}.",
            f"Prepare motor-pump teams for urban drains in {district}; deploy 4x HP submersibles per slum cluster.",
            f"Initiate phased evacuation of priority slum zones: {'; '.join(slums)}.",
            "Sanction temporary shelters in local schools + marriage halls with SDRF kitchen kits.",
            "Direct PWD to pre-clear major storm-water drains and desilt culverts under NH/SH.",
        ]
        agencies += [
            "TN Public Works Department (PWD) – Water Resources Organisation",
            "Tamil Nadu Fire and Rescue Services (TNFRS)",
            "TN State Disaster Response Force (TnSDRF) (swift-boat teams)",
            "Revenue Department (village-level ward volunteers)",
            "Greater Chennai Corporation / Municipal Administration (if applicable)",
        ]
        infra = [
            f"Pre-position 25+ HP motor pump sets at identified flood hubs in {district}",
            "Ready 35+ inflatable fibre rescue boats + 100 life jackets at taluk headquarters",
            f"Deploy 24x7 mobile health vans for {slums[0]} and 2 adjoining zones",
            "TANGEDCO — isolate low-tension feeders at flood-prone localities to prevent electrocution",
        ]

    elif disaster_type == "Cyclone":
        actions += [
            f"IMPACT PROBABILITY {risk_score*100:.0f}% — Issue DoT (Department of Telecom) Cell Broadcast + IGRS SMS to 30-km coastal buffer.",
            "IMMEDIATE FISHING EMBARGO: Order all mechanised & country craft back to harbour; deploy Fisheries Dept. patrols.",
            f"Activate COASTAL CYCLONE SHELTERS (CCCs) in {district}; open 72-hour supply stock (food/water/meds).",
            f"Alert {TANGEDCO_FEEDERS.get(district, 'TANGEDCO local circle')} — deploy line gangs + portable generators.",
            "Sandbag coastal bunds + tidal gates; begin precautionary sandbagging of low coast roads.",
            "Coordinate with NDRF for insertion of dog-squad + medical-paramedic teams at nearest Air Force node.",
        ]
        agencies += [
            "Indian National Centre for Ocean Information Services (INCOIS) liaison",
            "TN Fisheries Department (harbour & craft control)",
            "TN State Disaster Response Force (TnSDRF)",
            "NDRF Regional Response Center (Chennai / Arakkonam)",
            "TANGEDCO (Distribution and Transmission wings)",
            "TN Police — Coastal Security Group (CSG)",
            "Indian Coast Guard (rescue ops & maritime patrol)",
        ]
        if district in COASTAL_CYCLONE_DISTRICTS:
            actions.append(
                f"[{district} HIGH-EXPOSURE ZONE] — Mandate closure of schools/colleges; deploy 3 NDRF columns before landfall window."
            )
        infra = [
            f"Pre-stage 1000+ sandbags at {district} coastal taluk bunds",
            "Ready 50+ portable 125-kVA DG sets for shelter & primary health centres",
            "Pre-position 10 NDRF / TnSDRF rescue teams with inflatable boats and tree-cutting gear",
            "TANGEDCO — snap 11 kV coastal feeders at confirmed landfall approach",
        ]

    elif disaster_type == "Wildfire/Landslide":
        if district == "Nilgiris":
            actions += [
                f"SEVERE HILL HAZARD — Restrict traffic on {NILGIRI_HILL_ROADS[0]} and {NILGIRI_HILL_ROADS[1]}; permit only essential vehicles.",
                "Deploy earthmovers + rock-breakers at 5-km intervals on Mettupalayam-Ooty ghat; standby CRMP bulldozer squad at Coonoor.",
                "EVACUATE hutment clusters and tea-estate line-houses on >30° slopes (Coonoor, Kotagiri, Kundah taluks).",
                "Activate Nilgiris District Search & Rescue (NDSR) with mountaineering ropes + avalanche transceivers.",
            ]
            infra = [
                "Deploy 3x JCB 3DX + 2x CAT excavators at Coonoor, Kotagiri, Gudalur yard",
                "Position 8x portable weather stations along critical ghat sections",
                "Stock 500+ sandbags + 200 geotextile bags for slope stabilization at Kundah",
            ]
        else:
            actions += [
                "Issue fire-danger advisory to all forest beats; deploy fire watchers at high-fuel-load compartments.",
                "Ready TNFRS forest-fire tenders with high-pressure hoses + fire-beaters at main entry roads.",
                "Ban open burning and crop-stubble fires within 2 km of notified forest boundaries.",
                "Alert Revenue / Forest dept. mobile squads to patrol slope-facing hamlets for minor slips.",
            ]
            infra = [
                "Pre-stage 4x TNFRS forest tenders + 2x water tankers (10 kL) at range headquarters",
                "Deploy 2x JCB excavators at high-risk landslide zones within 10 km of major SH roads",
            ]
        actions.append(
            "Coordinate with TN Forest Dept. (GIS Cell) for live fire-perimeter & landslide-incident mapping."
        )
        agencies += [
            "TN Forest Department (Fire & Ecology wings)",
            "TN Highways Department (landslide clearance & slope stabilization)",
            "Tamil Nadu Fire and Rescue Services (TNFRS)",
            "Geological Survey of India (GSI) — landslide advisory liaison",
            "TnSDRF (high-altitude rescue teams, if applicable)",
        ]
        helplines["TN Forest Dept. Fire Line"] = "1800-425-7092"

    else:
        actions.append("General hazard advisory — maintain vigil.")
        infra = []

    cleaned_actions = [str(a).strip() for a in actions if str(a).strip()]
    return {
        "severity": severity,
        "alert_title": f"⚠️ {disaster_type.upper()} {severity.upper()} — {district}",
        "summary": (
            f"Risk confidence {risk_score*100:.0f}% for {disaster_type} in {district}. "
            f"Severity={severity}. DEOC must acknowledge and disseminate to all taluk control rooms within 30 minutes."
        ),
        "key_actions": list(dict.fromkeys(cleaned_actions))[:8],
        "agencies": list(dict.fromkeys(agencies)),
        "helplines": helplines,
        "infra": infra,
    }


if __name__ == "__main__":
    import sys, os
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    import json
    test_cases = [
        ("Chennai", "Flood", 0.82),
        ("Cuddalore", "Cyclone", 0.90),
        ("Nilgiris", "Wildfire/Landslide", 0.75),
        ("Madurai", "Wildfire/Landslide", 0.55),
        ("Thiruvallur", "Safe", 0.10),
    ]
    for d, t, s in test_cases:
        rec = generate_tn_recommendations(d, t, s)
        safe = lambda s: s.encode("ascii", "replace").decode("ascii")
        print("=" * 60)
        print(safe(rec["alert_title"]))
        print(safe(rec["summary"]))
        print("ACTIONS:")
        for a in rec["key_actions"]:
            print("  -", safe(a))
        print("INFRA:")
        for i in rec["infra"]:
            print("  *", safe(i))
        print()
