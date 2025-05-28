
def predict(component):
    # Dummy classifier: classify based on timing pattern
    if len(component) == 0:
        return "No signal"
    avg_time = sum(component) / len(component)
    if avg_time < 1.5:
        return "Species A"
    elif avg_time < 3.0:
        return "Species B"
    else:
        return "Species C"
