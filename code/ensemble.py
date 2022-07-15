import os

import pandas as pd


def main():
    print("Ensemble..: Start")
    dfs = [
        pd.read_csv(f"output/{df_file}") for df_file in os.listdir("output/")
    ]
    df_1, num_dfs = dfs[0], len(dfs)

    ensembled_order = []
    for idx in range(len(df_1)):
        ensembled_sample = {
            k: v / num_dfs  # TBD weighted based on performance?
            for v, k in enumerate(df_1.iloc[idx]["cell_order"].split(" "))
        }
        for df in dfs[1:]:
            sample = {
                k: v / num_dfs
                for v, k in enumerate(df.iloc[idx]["cell_order"].split(" "))
            }
            for key in ensembled_sample:
                ensembled_sample[key] += sample[key]
        ensembled_order.append(
            " ".join(
                [
                    i[0]
                    for i in list(
                        sorted(ensembled_sample.items(), key=lambda x: x[1])
                    )
                ]
            )
        )
    df_1["cell_order"] = ensembled_order

    df_1.to_csv("output/submission.csv", index=False)
    print("Ensemble..: Done!")


if __name__ == "__main__":
    main()
