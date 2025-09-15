Repository 3: 6G-TC ns-O-RAN Dataset Generator 
This repository provides our ns-O-RAN–based simulation framework for generating slice-aware datasets used to train ML models for Traffic Classification (TC) and Abnormal Behavior Detection (ABD).
Since testbeds such as OAI and Eurecom do not natively support full slicing or abnormal condition simulation, we extended the ns-3 + ns-O-RAN modules to model eMBB, URLLC, and mMTC slices, while injecting anomalies such as degraded link quality, fluctuating traffic load, and misconfigurations.
The generated datasets capture multi-layer KPIs (RRC, MAC, PHY) across slices and users, forming the backbone of our ML pipeline.
Key Features:
• Simulation of 3GPP-aligned RAN stack with O-RAN E2 interface.
• Configurable slices (eMBB, URLLC, mMTC) with heterogeneous traffic.
• Generation of labeled datasets for ML model training.
• Support for anomaly injection to develop robust ABD frameworks.
