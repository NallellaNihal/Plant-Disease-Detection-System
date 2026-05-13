RECOMMENDATIONS = {
    "Healthy": "Plant is healthy. Continue regular monitoring and watering.",

    "Leaf_Blight": "Remove infected leaves and apply fungicide. Avoid excessive moisture.",

    "Pepper__bell___Bacterial_spot":
    "Use copper-based bactericides and remove infected leaves immediately."
}
def get_recommendation(class_name: str) -> str:
    return RECOMMENDATIONS.get(
        class_name,
        "Consult an agriculture expert for proper treatment and prevention."
    )
