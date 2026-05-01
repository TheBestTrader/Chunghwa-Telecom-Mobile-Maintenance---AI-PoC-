import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# 時間軸：2023-12-31 23:45 到 2024-01-01 00:15，每分鐘一筆
start_time = datetime(2023, 12, 31, 23, 45, 0)
end_time = datetime(2024, 1, 1, 0, 15, 0)
timestamps = []
current = start_time
while current <= end_time:
    timestamps.append(current)
    current += timedelta(minutes=1)

n = len(timestamps)

# 壅塞程度曲線：0=正常, 1=最嚴重
# 23:55 開始上升，00:00 達到高峰，00:05 後逐漸緩解
def congestion_level(ts):
    # 轉換為相對分鐘數（以 23:45 為基準）
    base = datetime(2023, 12, 31, 23, 45, 0)
    minutes = (ts - base).total_seconds() / 60

    # 23:55 = 10 分鐘，00:00 = 15 分鐘，00:05 = 20 分鐘，00:15 = 30 分鐘
    if minutes < 10:       # 23:45 ~ 23:54：正常
        return 0.0
    elif minutes < 15:     # 23:55 ~ 23:59：快速上升
        return (minutes - 10) / 5 * 0.95
    elif minutes < 20:     # 00:00 ~ 00:04：高峰維持
        return 0.95 + np.random.uniform(-0.03, 0.03)
    elif minutes < 30:     # 00:05 ~ 00:14：逐漸緩解
        decay = (minutes - 20) / 10
        return max(0.0, 0.95 * (1 - decay) + np.random.uniform(-0.05, 0.05))
    else:
        return 0.0

congestion = np.array([congestion_level(ts) for ts in timestamps])

# PRB_Utilization：正常 ~45-65%，壅塞時逼近 100%
prb_base = np.random.uniform(45, 65, n)
prb_noise = np.random.normal(0, 2, n)
PRB_Utilization = np.clip(prb_base + congestion * 38 + prb_noise, 10, 99.9).round(1)

# RRC_Setup_Success_Rate：正常 ~97-99.5%，壅塞時暴跌至 55-65%
rrc_base = np.random.uniform(97, 99.5, n)
rrc_noise = np.random.normal(0, 0.5, n)
RRC_Setup_Success_Rate = np.clip(rrc_base - congestion * 42 + rrc_noise, 50, 100).round(2)

# Handover_Failure_Rate：正常 ~0.5-2%，壅塞時升高至 12-20%
ho_base = np.random.uniform(0.5, 2.0, n)
ho_noise = np.random.normal(0, 0.3, n)
Handover_Failure_Rate = np.clip(ho_base + congestion * 18 + ho_noise, 0, 25).round(2)

# System_Log：依壅塞程度隨機選取告警訊息
alarm_messages = [
    "Cell Congestion Alarm",
    "Radio Link Failure",
    "High Interference",
    "Throughput Degradation Warning",
    "Overload Control Activated",
]

def pick_log(cong_lvl):
    if cong_lvl < 0.15:
        return "System Normal"
    elif cong_lvl < 0.40:
        # 輕度壅塞：偶發告警
        return np.random.choice(["System Normal", alarm_messages[0], alarm_messages[2]],
                                p=[0.6, 0.25, 0.15])
    else:
        # 嚴重壅塞：高機率告警
        return np.random.choice(alarm_messages, p=[0.35, 0.25, 0.20, 0.12, 0.08])

System_Log = [pick_log(c) for c in congestion]

df = pd.DataFrame({
    "Timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
    "PRB_Utilization": PRB_Utilization,
    "RRC_Setup_Success_Rate": RRC_Setup_Success_Rate,
    "Handover_Failure_Rate": Handover_Failure_Rate,
    "System_Log": System_Log,
})

output_path = "network_mock_data.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"已生成 {len(df)} 筆模擬數據，儲存至 {output_path}")
print("\n--- 數據預覽 ---")
print(df.to_string(index=False))
