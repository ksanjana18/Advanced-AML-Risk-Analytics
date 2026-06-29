def get_risk_status(score):

    if score >= 70:
        return "Needs Review"

    elif score >= 40:
        return "Monitor"

    else:
        return "Low Risk"