#!/usr/bin/env python3

import argparse
from datetime import datetime
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_pandas(n_rows: int = 20) -> pd.DataFrame:
    """Create a sample DataFrame with data inconsistencies for testing cleanup agents.

    Args:
        n_rows (int): Number of rows to generate in the DataFrame (minimum 20)

    Returns:
        DataFrame: A DataFrame with specified number of rows and various data inconsistencies.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    if n_rows < 20:
        raise ValueError("Number of rows must be at least 20")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate IDs
    ids = list(range(1, n_rows + 1))

    # Generate dates with one inconsistent format
    base_date = datetime.now() - timedelta(days=365)
    dates = [
        (base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_rows)
    ]
    # Make one date inconsistent (European format) at random position
    inconsistent_date_idx = np.random.randint(0, n_rows)
    dates[inconsistent_date_idx] = "15/03/2023"

    # Generate fullnames with one incorrect format
    first_names = [
        "John", "Jane", "Michael", "Sarah", "David", "Emma", "Robert", "Lisa", "William", "Emily",
        "James", "Jennifer", "Thomas", "Mary", "Daniel", "Patricia", "Joseph", "Linda", "Charles", "Barbara",
        "Christopher", "Jessica", "Matthew", "Ashley", "Andrew", "Amanda", "Brian", "Stephanie", "Kevin", "Nicole",
        "Jason", "Heather", "Eric", "Elizabeth", "Adam", "Megan", "Steven", "Lauren", "Timothy", "Rachel",
        "Jeffrey", "Kimberly", "Ryan", "Christina", "Jacob", "Crystal", "Gary", "Michelle", "Nicholas", "Tiffany",
        "Jonathan", "Melissa", "Stephen", "Amber", "Larry", "Danielle", "Justin", "Brittany", "Scott", "Rebecca",
        "Brandon", "Laura", "Benjamin", "Emily", "Samuel", "Megan", "Gregory", "Hannah", "Alexander", "Samantha",
        "Frank", "Katherine", "Patrick", "Alexis", "Raymond", "Victoria", "Jack", "Madison", "Dennis", "Natalie",
        "Jerry", "Stephanie", "Tyler", "Courtney", "Aaron", "Kelly", "Jose", "Erin", "Henry", "Anna",
        "Douglas", "Sara", "Peter", "Vanessa", "Zachary", "Jasmine", "Walter", "Julia", "Alan", "Kelsey"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
        "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
        "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
        "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker", "Cruz", "Edwards", "Collins", "Reyes",
        "Stewart", "Morris", "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper",
        "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
        "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
        "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross", "Foster", "Jimenez"
    ]

    # Extend the name lists if more rows are needed
    if n_rows > 20:
        first_names = np.random.choice(first_names, n_rows, replace=True)
        last_names = np.random.choice(last_names, n_rows, replace=True)

    fullnames = [f"{first} {last}" for first, last in zip(first_names, last_names)]
    # Make one fullname incorrect (missing space) at random position
    inconsistent_name_idx = np.random.randint(0, n_rows)
    fullnames[inconsistent_name_idx] = "JamesWilson"

    # Generate emails with one duplicate
    emails = [
        f"{first.lower()}.{last.lower()}@example.com"
        for first, last in zip(first_names, last_names)
    ]
    # Make one email duplicate at random position
    inconsistent_email_idx = np.random.randint(0, n_rows)
    duplicate_email_idx = np.random.randint(0, n_rows)
    while duplicate_email_idx == inconsistent_email_idx:
        duplicate_email_idx = np.random.randint(0, n_rows)
    emails[inconsistent_email_idx] = emails[duplicate_email_idx]

    # Generate average_order_value with mixed currencies
    avg_order_values = np.random.uniform(10, 200, n_rows).round(2).astype(str)
    # Convert some to string with currency symbols at random positions
    currency_indices = np.random.choice(n_rows, 3, replace=False)
    avg_order_values[currency_indices[0]] = f"€{avg_order_values[currency_indices[0]]}"
    avg_order_values[currency_indices[1]] = f"£{avg_order_values[currency_indices[1]]}"
    avg_order_values[currency_indices[2]] = f"${avg_order_values[currency_indices[2]]}"

    # Generate number_of_purchases with one incorrect type
    num_purchases = np.random.randint(1, 20, n_rows)
    # Make one incorrect type (string instead of integer) at random position
    inconsistent_purchase_idx = np.random.randint(0, n_rows)
    num_purchases[inconsistent_purchase_idx] = 0.5

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "date": dates,
            "fullname": fullnames,
            "email": emails,
            "average_order_value": avg_order_values,
            "number_of_purchases": num_purchases,
        }
    )

    return df


def generate_sample_file(num_rows: int = 20, output_format: str = "csv") -> str:
    """
    Generate a sample data file with the specified number of rows and format.

    Args:
        num_rows (int): Number of rows to generate (default: 20)
        output_format (str): Output format, either 'csv' or 'excel' (default: 'csv')

    Returns:
        str: Path to the generated file
    """
    # Generate the DataFrame
    df = generate_pandas(num_rows)

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp and row count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sample-{num_rows}rows-{timestamp}"

    # Save the file based on the specified format
    if output_format.lower() == "excel":
        file_path = os.path.join(output_dir, f"{filename}.xlsx")
        df.to_excel(file_path, index=False)
    else:  # default to CSV
        file_path = os.path.join(output_dir, f"{filename}.csv")
        df.to_csv(file_path, index=False)

    return file_path


def main():
    parser = argparse.ArgumentParser(description="Generate sample data files")
    parser.add_argument(
        "--rows", type=int, default=20, help="Number of rows to generate (default: 20)"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "excel"],
        default="csv",
        help="Output format (default: csv)",
    )

    args = parser.parse_args()

    try:
        file_path = generate_sample_file(args.rows, args.format)
        print(f"Successfully generated sample file: {file_path}")
    except Exception as e:
        print(f"Error generating sample file: {str(e)}")


if __name__ == "__main__":
    main()
