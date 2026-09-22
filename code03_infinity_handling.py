import pandas as pd
import numpy as np


def main():
    file_path = "Supplementary_Data_Raw_and_Processed_Datasets.xlsx"
    sheet_name = "Sheet8_deal4"

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    print("Infinity correction using a fixed physical limit")

    numeric_columns = df.select_dtypes(include=np.number).columns
    physical_max_constant = 999.0

    for column in numeric_columns:
        if np.isinf(df[column]).any():
            infinity_count = np.isinf(df[column]).sum()

            df[column] = df[column].replace(
                np.inf,
                physical_max_constant,
            )
            df[column] = df[column].replace(
                -np.inf,
                -physical_max_constant,
            )

            print(
                f"- Feature [{column}] contains "
                f"{infinity_count} infinite values"
            )
            print(
                "  Replaced with fixed limits: "
                f"+/-{physical_max_constant}"
            )

    output_sheet = "Sheet9_deal5"
    with pd.ExcelWriter(
        file_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        df.to_excel(
            writer,
            sheet_name=output_sheet,
            index=False,
        )

    print(
        "Infinity correction completed. "
        f"Data saved to {output_sheet}"
    )


if __name__ == "__main__":
    main()
