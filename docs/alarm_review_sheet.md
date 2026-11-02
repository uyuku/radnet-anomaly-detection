# Clean-Background Test Alarm Audit: Author Review Sheet

This review sheet contains a stratified random sample of $N = 100$ clean-background alarms raised during the 2023–2025 held-out test split across the five pilot stations.
Each alarm includes synchronous multi-channel scintillation spectrometry, surface meteorology, and step continuity context.

## Categorization Guidelines for the Author
Please inspect each alarm's diagnostic plot (`reports/figures/alarm_review/alarm_ALM-XXX.png`) and assign one of the following candidate categories in the `assigned_category` column, along with a confidence rating (1 = Low to 5 = High) and explanatory rationale:

1. `rain_coincident`: Active rainfall at or immediately preceding the alarm ($P_{\text{1h}} \ge 0.1\text{ mm/h}$), accompanied by characteristic radon progeny gamma surges ($^{214}\text{Pb}$ and $^{214}\text{Bi}$).
2. `post_rain_washout`: Dry at the alarm timestamp ($P_{\text{1h}} = 0\text{ mm}$), but occurring within 6 hours of rain cessation with residual elevated $^{214}\text{Bi}$ ($R_{05}$) and $^{214}\text{Pb}$ ($R_{03}$).
3. `filter_change_artifact`: Discontinuity coincident with particulate filter tape advancement or maintenance cycle (sharp baseline step $\ge 25\%$).
4. `instrument_step_drift`: Sensor hardware modification, phototube gain adjustment, or permanent baseline step (e.g., Tampa post-March 2023 PMT replacement).
5. `other_natural`: Non-precipitation atmospheric phenomena, such as strong nocturnal temperature inversions trapping terrestrial radon, or barometric pressure front passages.
6. `unexplained`: Pure statistical counting fluctuation or Poisson noise without identifiable environmental or technical cause.

## Review Table

| Alarm ID | Station | Timestamp (UTC) | Gross CPM | Rain (mm/h) | Triggering Models | Diagnostic Plot | Assigned Category | Confidence (1-5) | Author Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- | :---: | :--- |
| ALM-001 | San Diego, CA | 2025-07-18 00:00:00 | 6646 | 0.0 | Tier 4 Nuclear | [ALM-001 Plot](../reports/figures/alarm_review/alarm_ALM-001.png) | | | |
| ALM-002 | Washington, DC | 2025-11-23 00:00:00 | 2100 | 0.0 | Tier 4 Nuclear | [ALM-002 Plot](../reports/figures/alarm_review/alarm_ALM-002.png) | | | |
| ALM-003 | Washington, DC | 2025-09-30 10:00:00 | 2739 | 0.0 | Tier 4 Nuclear | [ALM-003 Plot](../reports/figures/alarm_review/alarm_ALM-003.png) | | | |
| ALM-004 | Tampa, FL | 2023-12-02 14:00:00 | 1775 | 0.0 | Tier 4 Nuclear | [ALM-004 Plot](../reports/figures/alarm_review/alarm_ALM-004.png) | | | |
| ALM-005 | Tampa, FL | 2023-05-22 12:00:00 | 2154 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-005 Plot](../reports/figures/alarm_review/alarm_ALM-005.png) | | | |
| ALM-006 | Tampa, FL | 2023-10-07 14:00:00 | 1699 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-006 Plot](../reports/figures/alarm_review/alarm_ALM-006.png) | | | |
| ALM-007 | San Diego, CA | 2025-05-22 19:00:00 | 6842 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-007 Plot](../reports/figures/alarm_review/alarm_ALM-007.png) | | | |
| ALM-008 | San Diego, CA | 2025-07-09 02:00:00 | 6715 | 0.0 | Tier 4 Nuclear | [ALM-008 Plot](../reports/figures/alarm_review/alarm_ALM-008.png) | | | |
| ALM-009 | Tampa, FL | 2025-03-14 14:00:00 | 2892 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-009 Plot](../reports/figures/alarm_review/alarm_ALM-009.png) | | | |
| ALM-010 | Tampa, FL | 2023-05-11 14:00:00 | 1908 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-010 Plot](../reports/figures/alarm_review/alarm_ALM-010.png) | | | |
| ALM-011 | Tampa, FL | 2024-04-24 10:00:00 | 1861 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-011 Plot](../reports/figures/alarm_review/alarm_ALM-011.png) | | | |
| ALM-012 | Tampa, FL | 2023-01-30 09:00:00 | 1802 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-012 Plot](../reports/figures/alarm_review/alarm_ALM-012.png) | | | |
| ALM-013 | Tampa, FL | 2024-02-17 13:00:00 | 1889 | 0.0 | Tier 4 Nuclear | [ALM-013 Plot](../reports/figures/alarm_review/alarm_ALM-013.png) | | | |
| ALM-014 | San Diego, CA | 2025-08-08 22:00:00 | 6669 | 0.0 | Tier 4 Nuclear | [ALM-014 Plot](../reports/figures/alarm_review/alarm_ALM-014.png) | | | |
| ALM-015 | San Diego, CA | 2025-08-05 20:00:00 | 6745 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-015 Plot](../reports/figures/alarm_review/alarm_ALM-015.png) | | | |
| ALM-016 | Tampa, FL | 2024-05-01 05:00:00 | 1678 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-016 Plot](../reports/figures/alarm_review/alarm_ALM-016.png) | | | |
| ALM-017 | San Diego, CA | 2025-07-05 22:00:00 | 6624 | 0.0 | Tier 4 Nuclear | [ALM-017 Plot](../reports/figures/alarm_review/alarm_ALM-017.png) | | | |
| ALM-018 | San Diego, CA | 2025-07-25 23:00:00 | 6661 | 0.0 | Tier 4 Nuclear | [ALM-018 Plot](../reports/figures/alarm_review/alarm_ALM-018.png) | | | |
| ALM-019 | Tampa, FL | 2023-12-02 09:00:00 | 2313 | 0.0 | Tier 4 Nuclear; Rolling 3-sigma; Ro... | [ALM-019 Plot](../reports/figures/alarm_review/alarm_ALM-019.png) | | | |
| ALM-020 | Tampa, FL | 2023-10-07 08:00:00 | 1647 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-020 Plot](../reports/figures/alarm_review/alarm_ALM-020.png) | | | |
| ALM-021 | Washington, DC | 2025-09-05 00:00:00 | 2283 | 0.0 | Tier 4 Nuclear | [ALM-021 Plot](../reports/figures/alarm_review/alarm_ALM-021.png) | | | |
| ALM-022 | San Diego, CA | 2024-11-07 00:00:00 | 6871 | 0.0 | Tier 4 Nuclear | [ALM-022 Plot](../reports/figures/alarm_review/alarm_ALM-022.png) | | | |
| ALM-023 | Washington, DC | 2025-12-08 10:00:00 | 2602 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-023 Plot](../reports/figures/alarm_review/alarm_ALM-023.png) | | | |
| ALM-024 | Washington, DC | 2025-09-24 20:00:00 | 2540 | 0.0 | Tier 4 Nuclear | [ALM-024 Plot](../reports/figures/alarm_review/alarm_ALM-024.png) | | | |
| ALM-025 | San Diego, CA | 2024-10-24 01:00:00 | 6968 | 0.0 | Tier 4 Nuclear | [ALM-025 Plot](../reports/figures/alarm_review/alarm_ALM-025.png) | | | |
| ALM-026 | Tampa, FL | 2023-10-07 19:00:00 | 1327 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-026 Plot](../reports/figures/alarm_review/alarm_ALM-026.png) | | | |
| ALM-027 | Tampa, FL | 2023-09-27 03:00:00 | 1566 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-027 Plot](../reports/figures/alarm_review/alarm_ALM-027.png) | | | |
| ALM-028 | Tampa, FL | 2023-08-04 20:00:00 | 1339 | 0.0 | Tier 3 WeatherFusion | [ALM-028 Plot](../reports/figures/alarm_review/alarm_ALM-028.png) | | | |
| ALM-029 | Tampa, FL | 2023-04-06 08:00:00 | 1642 | 0.0 | Tier 3 WeatherFusion | [ALM-029 Plot](../reports/figures/alarm_review/alarm_ALM-029.png) | | | |
| ALM-030 | Tampa, FL | 2023-11-23 13:00:00 | 1875 | 0.0 | Tier 3 WeatherFusion; Rolling 3-sig... | [ALM-030 Plot](../reports/figures/alarm_review/alarm_ALM-030.png) | | | |
| ALM-031 | Tampa, FL | 2023-06-25 20:00:00 | 1343 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-031 Plot](../reports/figures/alarm_review/alarm_ALM-031.png) | | | |
| ALM-032 | Tampa, FL | 2023-06-25 18:00:00 | 1312 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-032 Plot](../reports/figures/alarm_review/alarm_ALM-032.png) | | | |
| ALM-033 | Tampa, FL | 2024-05-02 07:00:00 | 1921 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-033 Plot](../reports/figures/alarm_review/alarm_ALM-033.png) | | | |
| ALM-034 | San Diego, CA | 2025-05-31 19:00:00 | 7239 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-034 Plot](../reports/figures/alarm_review/alarm_ALM-034.png) | | | |
| ALM-035 | Tampa, FL | 2024-07-19 11:00:00 | 1749 | 0.0 | Tier 3 WeatherFusion | [ALM-035 Plot](../reports/figures/alarm_review/alarm_ALM-035.png) | | | |
| ALM-036 | Tampa, FL | 2024-08-20 16:00:00 | 1310 | 1.7 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-036 Plot](../reports/figures/alarm_review/alarm_ALM-036.png) | | | |
| ALM-037 | Tampa, FL | 2024-03-05 04:00:00 | 1581 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-037 Plot](../reports/figures/alarm_review/alarm_ALM-037.png) | | | |
| ALM-038 | Tampa, FL | 2023-03-06 12:00:00 | 1483 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-038 Plot](../reports/figures/alarm_review/alarm_ALM-038.png) | | | |
| ALM-039 | Tampa, FL | 2024-06-17 09:00:00 | 1525 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-039 Plot](../reports/figures/alarm_review/alarm_ALM-039.png) | | | |
| ALM-040 | Tampa, FL | 2024-04-27 23:00:00 | 1351 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-040 Plot](../reports/figures/alarm_review/alarm_ALM-040.png) | | | |
| ALM-041 | Tampa, FL | 2023-12-28 04:00:00 | 1706 | 0.3 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-041 Plot](../reports/figures/alarm_review/alarm_ALM-041.png) | | | |
| ALM-042 | Tampa, FL | 2024-07-13 12:00:00 | 1625 | 0.0 | Tier 3 WeatherFusion | [ALM-042 Plot](../reports/figures/alarm_review/alarm_ALM-042.png) | | | |
| ALM-043 | Tampa, FL | 2023-06-21 11:00:00 | 1495 | 2.3 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-043 Plot](../reports/figures/alarm_review/alarm_ALM-043.png) | | | |
| ALM-044 | Tampa, FL | 2023-05-06 08:00:00 | 1650 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-044 Plot](../reports/figures/alarm_review/alarm_ALM-044.png) | | | |
| ALM-045 | Tampa, FL | 2023-07-27 03:00:00 | 1612 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-045 Plot](../reports/figures/alarm_review/alarm_ALM-045.png) | | | |
| ALM-046 | Tampa, FL | 2023-01-11 11:00:00 | 1644 | 0.0 | Tier 3 WeatherFusion | [ALM-046 Plot](../reports/figures/alarm_review/alarm_ALM-046.png) | | | |
| ALM-047 | Tampa, FL | 2023-08-23 08:00:00 | 1579 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-047 Plot](../reports/figures/alarm_review/alarm_ALM-047.png) | | | |
| ALM-048 | Tampa, FL | 2023-05-14 08:00:00 | 2048 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-048 Plot](../reports/figures/alarm_review/alarm_ALM-048.png) | | | |
| ALM-049 | Tampa, FL | 2023-03-28 12:00:00 | 1937 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-049 Plot](../reports/figures/alarm_review/alarm_ALM-049.png) | | | |
| ALM-050 | Tampa, FL | 2024-06-26 18:00:00 | 1191 | 0.0 | Tier 3 WeatherFusion | [ALM-050 Plot](../reports/figures/alarm_review/alarm_ALM-050.png) | | | |
| ALM-051 | San Diego, CA | 2025-11-08 17:00:00 | 8989 | 0.0 | Global 3-sigma | [ALM-051 Plot](../reports/figures/alarm_review/alarm_ALM-051.png) | | | |
| ALM-052 | San Diego, CA | 2023-04-07 16:00:00 | 6968 | 0.0 | Rolling 3-sigma | [ALM-052 Plot](../reports/figures/alarm_review/alarm_ALM-052.png) | | | |
| ALM-053 | San Diego, CA | 2023-06-26 11:00:00 | 7840 | 0.0 | Rolling 3-sigma | [ALM-053 Plot](../reports/figures/alarm_review/alarm_ALM-053.png) | | | |
| ALM-054 | San Diego, CA | 2025-05-21 14:00:00 | 8411 | 0.0 | Rolling 3-sigma; Global 3-sigma | [ALM-054 Plot](../reports/figures/alarm_review/alarm_ALM-054.png) | | | |
| ALM-055 | San Diego, CA | 2023-09-14 22:00:00 | 9257 | 0.0 | Tier 4 Nuclear; Rolling 3-sigma; Ro... | [ALM-055 Plot](../reports/figures/alarm_review/alarm_ALM-055.png) | | | |
| ALM-056 | Washington, DC | 2023-05-21 01:00:00 | 3178 | 3.0 | Rolling 3-sigma; Rolling 4-sigma; G... | [ALM-056 Plot](../reports/figures/alarm_review/alarm_ALM-056.png) | | | |
| ALM-057 | San Diego, CA | 2024-11-11 16:00:00 | 9013 | 0.0 | Global 3-sigma | [ALM-057 Plot](../reports/figures/alarm_review/alarm_ALM-057.png) | | | |
| ALM-058 | San Diego, CA | 2025-11-02 14:00:00 | 8903 | 0.0 | Global 3-sigma | [ALM-058 Plot](../reports/figures/alarm_review/alarm_ALM-058.png) | | | |
| ALM-059 | Birmingham, AL | 2023-08-24 11:00:00 | 5117 | 0.0 | Global 3-sigma | [ALM-059 Plot](../reports/figures/alarm_review/alarm_ALM-059.png) | | | |
| ALM-060 | San Diego, CA | 2023-12-19 08:00:00 | 8500 | 0.0 | Rolling 3-sigma; Global 3-sigma | [ALM-060 Plot](../reports/figures/alarm_review/alarm_ALM-060.png) | | | |
| ALM-061 | San Diego, CA | 2024-09-18 08:00:00 | 6657 | 0.0 | Rolling 3-sigma; Rolling 4-sigma | [ALM-061 Plot](../reports/figures/alarm_review/alarm_ALM-061.png) | | | |
| ALM-062 | Birmingham, AL | 2024-06-22 13:00:00 | 5085 | 0.0 | Global 3-sigma | [ALM-062 Plot](../reports/figures/alarm_review/alarm_ALM-062.png) | | | |
| ALM-063 | San Diego, CA | 2025-02-01 14:00:00 | 8793 | 0.0 | Global 3-sigma | [ALM-063 Plot](../reports/figures/alarm_review/alarm_ALM-063.png) | | | |
| ALM-064 | Birmingham, AL | 2024-01-16 04:00:00 | 4780 | 0.0 | Global 3-sigma | [ALM-064 Plot](../reports/figures/alarm_review/alarm_ALM-064.png) | | | |
| ALM-065 | San Diego, CA | 2025-08-19 08:00:00 | 8306 | 0.0 | Rolling 3-sigma | [ALM-065 Plot](../reports/figures/alarm_review/alarm_ALM-065.png) | | | |
| ALM-066 | Tampa, FL | 2025-12-13 16:00:00 | 2632 | 0.0 | Rolling 3-sigma | [ALM-066 Plot](../reports/figures/alarm_review/alarm_ALM-066.png) | | | |
| ALM-067 | Birmingham, AL | 2023-01-03 21:00:00 | 4304 | 5.1 | Rolling 3-sigma; Rolling 4-sigma | [ALM-067 Plot](../reports/figures/alarm_review/alarm_ALM-067.png) | | | |
| ALM-068 | San Diego, CA | 2023-07-12 13:00:00 | 8638 | 0.0 | Rolling 3-sigma; Rolling 4-sigma; G... | [ALM-068 Plot](../reports/figures/alarm_review/alarm_ALM-068.png) | | | |
| ALM-069 | San Diego, CA | 2024-10-11 19:00:00 | 7385 | 0.0 | Rolling 3-sigma; Rolling 4-sigma | [ALM-069 Plot](../reports/figures/alarm_review/alarm_ALM-069.png) | | | |
| ALM-070 | San Diego, CA | 2025-08-20 14:00:00 | 8842 | 0.0 | Rolling 3-sigma; Global 3-sigma | [ALM-070 Plot](../reports/figures/alarm_review/alarm_ALM-070.png) | | | |
| ALM-071 | San Diego, CA | 2025-01-25 16:00:00 | 9501 | 0.0 | Rolling 3-sigma; Global 3-sigma | [ALM-071 Plot](../reports/figures/alarm_review/alarm_ALM-071.png) | | | |
| ALM-072 | Dallas, TX | 2023-01-24 20:00:00 | 4773 | 1.3 | Rolling 3-sigma; Rolling 4-sigma; G... | [ALM-072 Plot](../reports/figures/alarm_review/alarm_ALM-072.png) | | | |
| ALM-073 | Washington, DC | 2024-01-28 08:00:00 | 2896 | 3.8 | Rolling 3-sigma; Rolling 4-sigma; G... | [ALM-073 Plot](../reports/figures/alarm_review/alarm_ALM-073.png) | | | |
| ALM-074 | Birmingham, AL | 2023-11-07 16:00:00 | 4987 | 0.0 | Global 3-sigma | [ALM-074 Plot](../reports/figures/alarm_review/alarm_ALM-074.png) | | | |
| ALM-075 | San Diego, CA | 2024-12-23 15:00:00 | 8686 | 0.0 | Global 3-sigma | [ALM-075 Plot](../reports/figures/alarm_review/alarm_ALM-075.png) | | | |
| ALM-076 | San Diego, CA | 2023-12-05 07:00:00 | 8324 | 0.0 | Rolling 3-sigma | [ALM-076 Plot](../reports/figures/alarm_review/alarm_ALM-076.png) | | | |
| ALM-077 | Tampa, FL | 2023-12-28 05:00:00 | 1785 | 0.5 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-077 Plot](../reports/figures/alarm_review/alarm_ALM-077.png) | | | |
| ALM-078 | San Diego, CA | 2023-10-29 18:00:00 | 7040 | 0.0 | Rolling 3-sigma | [ALM-078 Plot](../reports/figures/alarm_review/alarm_ALM-078.png) | | | |
| ALM-079 | Birmingham, AL | 2024-07-27 21:00:00 | 6225 | 7.1 | Rolling 3-sigma; Rolling 4-sigma; G... | [ALM-079 Plot](../reports/figures/alarm_review/alarm_ALM-079.png) | | | |
| ALM-080 | San Diego, CA | 2024-06-15 10:00:00 | 7134 | 0.0 | Rolling 3-sigma; Rolling 4-sigma | [ALM-080 Plot](../reports/figures/alarm_review/alarm_ALM-080.png) | | | |
| ALM-081 | Tampa, FL | 2024-08-16 05:00:00 | 1335 | 0.0 | Tier 2 Spectral | [ALM-081 Plot](../reports/figures/alarm_review/alarm_ALM-081.png) | | | |
| ALM-082 | San Diego, CA | 2023-12-05 06:00:00 | 8236 | 0.0 | Rolling 3-sigma | [ALM-082 Plot](../reports/figures/alarm_review/alarm_ALM-082.png) | | | |
| ALM-083 | Tampa, FL | 2023-11-21 15:00:00 | 1499 | 0.0 | Tier 2 Spectral | [ALM-083 Plot](../reports/figures/alarm_review/alarm_ALM-083.png) | | | |
| ALM-084 | Tampa, FL | 2023-02-25 14:00:00 | 2052 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-084 Plot](../reports/figures/alarm_review/alarm_ALM-084.png) | | | |
| ALM-085 | Tampa, FL | 2023-01-02 13:00:00 | 2852 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-085 Plot](../reports/figures/alarm_review/alarm_ALM-085.png) | | | |
| ALM-086 | Tampa, FL | 2024-07-04 02:00:00 | 1625 | 0.0 | Tier 3 WeatherFusion; Tier 2 Spectr... | [ALM-086 Plot](../reports/figures/alarm_review/alarm_ALM-086.png) | | | |
| ALM-087 | Tampa, FL | 2024-05-09 06:00:00 | 1308 | 0.0 | Tier 2 Spectral | [ALM-087 Plot](../reports/figures/alarm_review/alarm_ALM-087.png) | | | |
| ALM-088 | Tampa, FL | 2023-07-03 13:00:00 | 1555 | 0.0 | Tier 2 Spectral | [ALM-088 Plot](../reports/figures/alarm_review/alarm_ALM-088.png) | | | |
| ALM-089 | Tampa, FL | 2023-06-27 04:00:00 | 1352 | 0.0 | Tier 2 Spectral | [ALM-089 Plot](../reports/figures/alarm_review/alarm_ALM-089.png) | | | |
| ALM-090 | Tampa, FL | 2023-12-16 22:00:00 | 1398 | 4.0 | Tier 2 Spectral | [ALM-090 Plot](../reports/figures/alarm_review/alarm_ALM-090.png) | | | |
| ALM-091 | San Diego, CA | 2024-06-06 15:00:00 | 7038 | 0.0 | Rolling 3-sigma | [ALM-091 Plot](../reports/figures/alarm_review/alarm_ALM-091.png) | | | |
| ALM-092 | Tampa, FL | 2023-12-12 20:00:00 | 1408 | 0.0 | Tier 2 Spectral | [ALM-092 Plot](../reports/figures/alarm_review/alarm_ALM-092.png) | | | |
| ALM-093 | Tampa, FL | 2024-04-27 15:00:00 | 1529 | 0.0 | Tier 2 Spectral | [ALM-093 Plot](../reports/figures/alarm_review/alarm_ALM-093.png) | | | |
| ALM-094 | Tampa, FL | 2023-08-04 23:00:00 | 2553 | 0.0 | Tier 4 Nuclear; Tier 3 WeatherFusio... | [ALM-094 Plot](../reports/figures/alarm_review/alarm_ALM-094.png) | | | |
| ALM-095 | Washington, DC | 2025-11-10 05:00:00 | 2637 | 0.0 | Tier 4 Nuclear | [ALM-095 Plot](../reports/figures/alarm_review/alarm_ALM-095.png) | | | |
| ALM-096 | San Diego, CA | 2025-07-23 21:00:00 | 6673 | 0.0 | Tier 4 Nuclear | [ALM-096 Plot](../reports/figures/alarm_review/alarm_ALM-096.png) | | | |
| ALM-097 | Tampa, FL | 2024-01-13 14:00:00 | 1437 | 0.0 | Tier 2 Spectral | [ALM-097 Plot](../reports/figures/alarm_review/alarm_ALM-097.png) | | | |
| ALM-098 | Tampa, FL | 2024-07-19 19:00:00 | 1267 | 0.0 | Tier 2 Spectral | [ALM-098 Plot](../reports/figures/alarm_review/alarm_ALM-098.png) | | | |
| ALM-099 | Tampa, FL | 2024-04-18 11:00:00 | 1471 | 0.0 | Tier 2 Spectral | [ALM-099 Plot](../reports/figures/alarm_review/alarm_ALM-099.png) | | | |
| ALM-100 | Tampa, FL | 2024-03-25 21:00:00 | 1332 | 0.0 | Tier 2 Spectral | [ALM-100 Plot](../reports/figures/alarm_review/alarm_ALM-100.png) | | | |
