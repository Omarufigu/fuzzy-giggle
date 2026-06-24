"""
Name: Omar Figueroa
Email: omar.figueroa890@myhunter.cuny.edu
Resources: Used course assignment handout and ChatGPT for code drafting.
"""

import pandas as pd


DATA_COLUMNS = [
    "dbn",
    "school_name",
    "NTA",
    "graduation_rate",
    "pct_stu_safe",
    "attendance_rate",
    "college_career_rate",
    "language_classes",
    "advancedplacement_courses",
    "method1",
    "overview_paragraph",
]

NUMERIC_COLUMNS = [
    "graduation_rate",
    "pct_stu_safe",
    "attendance_rate",
    "college_career_rate",
]


# pylint: disable=invalid-name

def import_data(file_name):
    """Read the school CSV, keep needed columns, and drop missing grad rates."""
    df = pd.read_csv(file_name)
    df = df[DATA_COLUMNS]
    df = df.dropna(subset=["graduation_rate"])

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def impute_numeric_cols(df):
    """Fill missing numeric school columns with each column's median."""
    new_df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
            new_df[col] = new_df[col].fillna(new_df[col].median())
    return new_df


def _count_items(entry):
    """Count comma-separated items in a cell."""
    if pd.isna(entry):
        return 0
    text = str(entry).strip()
    if text == "":
        return 0
    return len([item for item in text.split(",") if item.strip()])


def compute_count_col(df, col):
    """Return counts of comma-separated items in df[col]."""
    return df[col].apply(_count_items)


def encode_categorical_col(col):
    """One-hot encode comma-separated categorical entries into sorted columns."""
    values = set()
    for entry in col:
        if pd.isna(entry):
            continue
        for item in str(entry).split(","):
            item = item.strip()
            if item:
                values.add(item)

    sorted_values = sorted(values)
    data = {}
    for value in sorted_values:
        data[value] = col.apply(
            lambda entry, target=value: int(
                not pd.isna(entry)
                and target in [part.strip() for part in str(entry).split(",")]
            )
        )
    return pd.DataFrame(data, index=col.index)


def split_test_train(df, xes_col_names, y_col_name, frac=0.25,
                     random_state=922):
    """Split a DataFrame into train/test x and y sets."""
    df_test = df.sample(frac=frac, random_state=random_state)
    df_train = df.copy()
    df_train = df_train.drop(df_test.index)
    return (
        df_train[xes_col_names],
        df_test[xes_col_names],
        df_train[y_col_name],
        df_test[y_col_name],
    )


def compute_lin_reg(xes, yes):
    """Compute theta_0 and theta_1 for simple linear regression."""
    xes = pd.Series(xes, dtype="float64")
    yes = pd.Series(yes, dtype="float64")
    clean_df = pd.DataFrame({"xes": xes, "yes": yes}).dropna()
    xes = clean_df["xes"]
    yes = clean_df["yes"]

    sd_x = xes.std()
    sd_y = yes.std()
    if sd_x == 0 or pd.isna(sd_x):
        theta_1 = 0
    else:
        theta_1 = xes.corr(yes) * sd_y / sd_x
    theta_0 = yes.mean() - theta_1 * xes.mean()
    return theta_0, theta_1


def predict(xes, theta_0, theta_1):
    """Predict y values from x values using theta_0 + theta_1 * x."""
    return theta_0 + theta_1 * pd.Series(xes)


def mse_loss(y_actual, y_estimate):
    """Return mean squared error loss."""
    actual = pd.Series(y_actual, dtype="float64").reset_index(drop=True)
    estimate = pd.Series(y_estimate, dtype="float64").reset_index(drop=True)
    return ((actual - estimate) ** 2).mean()


def rmse_loss(y_actual, y_estimate):
    """Return root mean squared error loss."""
    return mse_loss(y_actual, y_estimate) ** 0.5


def compute_error(y_actual, y_estimate, loss_fnc=mse_loss):
    """Return the requested loss function applied to actual and estimate."""
    return loss_fnc(y_actual, y_estimate)


def count_lengths(entries):
    """Return a dictionary counting string lengths in entries."""
    counts = {}
    for entry in entries:
        length = 0 if pd.isna(entry) else len(str(entry))
        counts[length] = counts.get(length, 0) + 1
    return counts


def count_sentences(entries):
    """Return a dictionary counting periods in each entry."""
    counts = {}
    for entry in entries:
        sentence_count = 0 if pd.isna(entry) else str(entry).count(".")
        counts[sentence_count] = counts.get(sentence_count, 0) + 1
    return counts


def compute_mean(counts):
    """Compute the weighted mean of keys in a count dictionary."""
    total_count = sum(counts.values())
    if total_count == 0:
        return 0
    total_sum = sum(value * count for value, count in counts.items())
    return total_sum / total_count


def _series_equal(left, right):
    """Compare two Series by values, ignoring index labels."""
    left = pd.Series(left).reset_index(drop=True)
    right = pd.Series(right).reset_index(drop=True)
    return left.equals(right)


def test_compute_count_col(compute_fnc=compute_count_col):
    """Test whether compute_fnc correctly counts comma-separated items."""
    try:
        df = pd.DataFrame({
            "items": ["A, B, C", "A", None, "", "Red, Blue"],
        })
        expected = pd.Series([3, 1, 0, 0, 2])
        result = compute_fnc(df, "items")
        return _series_equal(result, expected)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def test_predict(predict_fnc=predict):
    """Test whether predict_fnc correctly applies a linear model."""
    try:
        xes = pd.Series([0, 1, 2, 3])
        expected = pd.Series([5, 7, 9, 11])
        result = predict_fnc(xes, 5, 2)
        return _series_equal(result, expected)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def test_mse_loss(loss_fnc=mse_loss):
    """Test whether loss_fnc correctly computes mean squared error."""
    try:
        actual = pd.Series([1, 2, 3, 4])
        estimate = pd.Series([1, 4, 1, 4])
        result = loss_fnc(actual, estimate)
        return abs(result - 2) < 0.0000001
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
