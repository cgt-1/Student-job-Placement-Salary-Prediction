def add_features(df):
    df = df.copy()

    df['academic_score'] = (
        df['cgpa'] +
        df['tenth_percentage']/10 +
        df['twelfth_percentage']/10
    ) / 3

    df['technical_score'] = (
        df['coding_skill_rating'] +
        df['aptitude_skill_rating'] +
        df['projects_completed'] +
        df['internships_completed']
    ) / 4

    return df