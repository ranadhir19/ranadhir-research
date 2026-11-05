# Field data — provenance (no raw data redistributed)

The field runs use three public datasets. Their raw files are not
redistributed in this deposit: the Olist and IBM sample-data terms
do not clearly permit redistribution, so the deposit carries this
provenance record and the analysis scripts instead. To reproduce
the field runs, download each dataset from its source, verify the
SHA-256 below, and place the files in this `field_data/` folder
(the scripts read them from here).

## Olist Brazilian E-Commerce Public Dataset (Olist, 2018)

Source: Kaggle, "Brazilian E-Commerce Public Dataset by Olist"
(also distributed via the Olist GitHub organisation). The dataset
is published under non-commercial terms on its Kaggle page; it is
not redistributed here. Files used:

| File | Rows (incl. header) | SHA-256 |
|---|---|---|
| `olist_customers_dataset.csv` | 99,442 | `c26c17f59f3027a6a0dbeb8f1fa373c38f3323ecc84f225a8f4820e5bb288df8` |
| `olist_order_items_dataset.csv` | 112,651 | `4f6abdbbc94036d0df4a76fa0520c072e31a40119d70f7f370fba1e2285d2bcb` |
| `olist_order_payments_dataset.csv` | 103,887 | `61674868ed7b872aac3dcd95bf774448a88f6426b183f21be55f75af021ffdec` |
| `olist_order_reviews_dataset.csv` | 104,720 | `012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7` |
| `olist_orders_dataset.csv` | 99,442 | `8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf` |

## IBM Telco Customer Churn sample data

Source: IBM community sample datasets ("Telco customer churn"),
file `telco_churn.csv` (the WA_Fn-UseC_-Telco-Customer-Churn
extract). Distributed by IBM as sample data for product
demonstration and learning; no open redistribution licence is
stated, so the file is not redistributed here.

| File | Rows (incl. header) | SHA-256 |
|---|---|---|
| `telco_churn.csv` | 7,044 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |

## UCI Bike Sharing Dataset (Fanaee-T & Gama, 2014)

Source: UCI Machine Learning Repository, "Bike Sharing" dataset.
Licence: CC BY 4.0 — redistribution is permitted with attribution,
and the extracted day-level file is therefore included in this
deposit at `upgrade_data/extracted/day.csv`. The field code reads
the original archive; place it at `upgrade_data/bike_sharing.zip`
to re-run.

| File | SHA-256 |
|---|---|
| `bike_sharing.zip` (original archive, not deposited) | `b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401` |
| `day.csv` (extracted, deposited; 732 rows incl. header) | `a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be` |

Citation for the dataset: Fanaee-T, Hadi, and Gama, Joao,
"Event labeling combining ensemble detectors and background
knowledge", Progress in Artificial Intelligence (2014).
