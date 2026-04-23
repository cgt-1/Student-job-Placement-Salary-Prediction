def handle_missing(df):
    df = df.copy()
    df['extracurricular_involvement'] = df['extracurricular_involvement'].fillna('Missing')
    return df


def handle_outliers(df):
    df = df.copy()

    cols = [
        'cgpa', 'tenth_percentage', 'twelfth_percentage',
        'study_hours_per_day', 'attendance_percentage',
        'sleep_hours', 'salary_lpa'
    ]

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        median = df[col].median()

        df.loc[df[col] < lower, col] = median
        df.loc[df[col] > upper, col] = median

    return df