
def get_monitoring_status(
    heart_rate,
    gsr_status,
    minutes_since_last_restroom,
    restroom_threshold=180
):

    alerts = []

    if heart_rate >= 100:
        alerts.append("Elevated heart rate")

    if gsr_status == "Elevated":
        alerts.append("Elevated GSR")

    if minutes_since_last_restroom >= restroom_threshold:
        alerts.append("Restroom reminder")

    return {
        "alert": len(alerts) > 0,
        "messages": alerts
    }
