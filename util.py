def getErrorCount(y_pred, y_test):
    zip_object = zip(y_pred, y_test)
    error_count = 0
    for i, j in zip_object:
        if i != j:
            error_count += 1

    print("Test Count = ", len(y_pred), "  Count = ", error_count)

    return error_count