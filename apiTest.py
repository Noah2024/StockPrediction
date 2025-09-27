from api import * 

# print(getLastKnownData("AAPL"))
# Initialize client
# getFinancialsLastReported("AAPL", "quarterly")
# getAllHistoric("AAPL")
data = finnhubClient.financials_reported(symbol="AAPL", freq="quarterly")
data = data["data"]
print(data[0])
print(len(data[0]))

# for I in data[0]:
#     print(I)
#     print(data[0][I])

def fin_report_json_to_csv_with_pct_change(json_data, csv_path):
    import pandas as pd

    rows = []
    all_concepts = set()
    
    for entry in json_data:
        row = {
            "accessNumber": entry.get("accessNumber"),
            "symbol": entry.get("symbol"),
            "year": entry.get("year"),
            "quarter": entry.get("quarter"),
            "form": entry.get("form"),
            "startDate": entry.get("startDate"),
            "endDate": entry.get("endDate"),
            "filedDate": entry.get("filedDate"),
            "acceptedDate": entry.get("acceptedDate"),
        }
        for section in ["bs", "ic", "cf"]:
            for item in entry.get("report", {}).get(section, []):
                concept = item.get("concept")
                value = item.get("value")
                if concept is not None:
                    row[concept] = value
                    all_concepts.add(concept)
        rows.append(row)
    
    columns = [
        "accessNumber", "symbol", "year", "quarter", "form",
        "startDate", "endDate", "filedDate", "acceptedDate"
    ] + sorted(all_concepts)
    df = pd.DataFrame(rows, columns=columns)
    
    # Find numeric columns (excluding metadata columns)
    meta_cols = [
        "accessNumber", "symbol", "year", "quarter", "form",
        "startDate", "endDate", "filedDate", "acceptedDate"
    ]
    numeric_cols = [col for col in df.columns if col not in meta_cols]
    
    # Convert numeric columns to float for percent change calculation
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Sort by year and quarter to ensure correct order
    df = df.sort_values(by=["year", "quarter"]).reset_index(drop=True)
    
    # Calculate percent change for each numeric column (1.0 = 100%)
    pct_change_df = df[numeric_cols].pct_change()
    pct_change_df.columns = [f"{col}_pct_change" for col in numeric_cols]
    

    # columns_to_keep = [
    # "symbol", "year", "quarter", "startDate","endDate","filedDate","acceptedDate", "us-gaap_InventoryNet","us-gaap_Liabilities","us-gaap_LiabilitiesAndStockholdersEquity","us-gaap_LiabilitiesCurrent","us-gaap_NetIncomeLoss","us-gaap_NonoperatingIncomeExpense","us-gaap_OperatingExpenses","us-gaap_OperatingIncomeLoss","us-gaap_OtherAssetsCurrent","us-gaap_OtherAssetsNoncurrent","us-gaap_OtherLiabilitiesNoncurrent","us-gaap_PaymentsForProceedsFromOtherInvestingActivities","us-gaap_ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities","us-gaap_ResearchAndDevelopmentExpense","us-gaap_RetainedEarningsAccumulatedDeficit","us-gaap_SellingGeneralAndAdministrativeExpense","us-gaap_ShareBasedCompensation","us-gaap_StockholdersEquity","us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding","us-gaap_WeightedAverageNumberOfSharesOutstandingBasic","us-gaap_AccountsPayableCurrent_pct_change","us-gaap_AccountsReceivableNetCurrent_pct_change","us-gaap_AccumulatedOtherComprehensiveIncomeLossNetOfTax_pct_change","us-gaap_Assets_pct_change","us-gaap_AssetsCurrent_pct_change","us-gaap_CashAndCashEquivalentsAtCarryingValue_pct_change","us-gaap_CostOfGoodsAndServicesSold_pct_change","us-gaap_EarningsPerShareBasic_pct_change","us-gaap_EarningsPerShareDiluted_pct_change","us-gaap_GrossProfit_pct_change","us-gaap_IncomeTaxExpenseBenefit_pct_change","us-gaap_IncomeTaxesPaidNet_pct_change","us-gaap_IncreaseDecreaseInAccountsPayable_pct_change","us-gaap_IncreaseDecreaseInAccountsReceivable_pct_change","us-gaap_IncreaseDecreaseInInventories_pct_change","us-gaap_IncreaseDecreaseInOtherOperatingAssets_pct_change","us-gaap_IncreaseDecreaseInOtherOperatingLiabilities_pct_change","us-gaap_InventoryNet_pct_change","us-gaap_Liabilities_pct_change","us-gaap_LiabilitiesAndStockholdersEquity_pct_change","us-gaap_LiabilitiesCurrent_pct_change","us-gaap_NetIncomeLoss_pct_change","us-gaap_NonoperatingIncomeExpense_pct_change","us-gaap_OperatingExpenses_pct_change","us-gaap_OperatingIncomeLoss_pct_change","us-gaap_OtherAssetsCurrent_pct_change","us-gaap_OtherAssetsNoncurrent_pct_change","us-gaap_OtherLiabilitiesNoncurrent_pct_change","us-gaap_PaymentsForProceedsFromOtherInvestingActivities_pct_change","us-gaap_ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities_pct_change","us-gaap_ResearchAndDevelopmentExpense_pct_change","us-gaap_RetainedEarningsAccumulatedDeficit_pct_change","us-gaap_SellingGeneralAndAdministrativeExpense_pct_change","us-gaap_ShareBasedCompensation_pct_change","us-gaap_StockholdersEquity_pct_change","us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding_pct_change","us-gaap_WeightedAverageNumberOfSharesOutstandingBasic_pct_change"
    # ]

    # Only keep columns that exist in the DataFrame
    # columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    # df = df[columns_to_keep]
    
    # Optionally drop rows with any missing values now
    df = pct_change_df.dropna(axis=0, how='any')
    
    df.to_csv(csv_path, index=False)

# pandasData = pd.DataFrame([data[0]["report"]])
# df.to_csv("AAPL_quarterly.csv", index=False)

fin_report_json_to_csv_with_pct_change(data, "AAPL_quarterly3.csv")

#usefulData#finnhubClient.financials_reported(symbol=ticker, freq=freq)