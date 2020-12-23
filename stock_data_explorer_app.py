import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

def main():
    
    #set title and sidebar introduction
   
    st.set_page_config(layout="centered")
    
    st.title("Stock Pricing and fundamental drivers by Sector")
    st.sidebar.title("Data Exploration Web App")
    st.sidebar.markdown("What is the Company Stock Pricing by sector, and how does it relate to fundamental financial drivers?")
    st.sidebar.markdown("Fundamental financial drivers shown are expected growth and expected margins for the next 10 years, as well as two risk measures, Beta of the past 5 years and current Net Debt as a % of Enterprise Value.")
     
    @st.cache(persist=True, allow_output_mutation=True)
    def load_data():
        data = pd.read_excel("company_data_new.xlsm")
        return data
    
    #load data
    df = load_data()
   
    #set columns to numeric
    df['EV_TTM_EBITDA'] = df['EV_TTM_EBITDA'].apply(pd.to_numeric, errors='coerce')
    df['EBITDA_Forecast_CAGR_10y'] = df['EBITDA_Forecast_CAGR_10y'].apply(pd.to_numeric, errors='coerce')
    df['Avg_EBITDA_Margin_Forecast_10y'] = df['Avg_EBITDA_Margin_Forecast_10y'].apply(pd.to_numeric, errors='coerce')
    df['Beta_5y'] = df['Beta_5y'].apply(pd.to_numeric, errors='coerce')
    df['Net_Debt_perc_EV'] = df['Net_Debt_perc_EV'].apply(pd.to_numeric, errors='coerce')
    
    df['Market_Cap_TTM_Net_Income'] = df['Market_Cap_TTM_Net_Income'].apply(pd.to_numeric, errors='coerce')
    df['Net_Income_Forecast_CAGR_10y'] = df['Net_Income_Forecast_CAGR_10y'].apply(pd.to_numeric, errors='coerce')
    df['Avg_Net_Income_Margin_Forecast_10y'] = df['Avg_Net_Income_Margin_Forecast_10y'].apply(pd.to_numeric, errors='coerce')
    
    #sidebar for price multiple selection
    st.sidebar.subheader("Choose Pricing Multiple")
    price_multiple_list = ['EV to EBITDA ratio', 
                       'Price to Earnings ratio']
    price_multiple = st.sidebar.selectbox("Select Pricing multiple", (price_multiple_list))
    
    #sidebar for sector selection
    st.sidebar.subheader("Choose Sector")
    
    sector_list = ['All']
    for sector in df['Sector'].unique():
        sector_list.append(sector)
    sector = st.sidebar.selectbox("Filter by company sector. Note Finaical sector not applicable for EV to EBITDA", (sector_list))
    
    #sidebar for selection of outlier removal threshold
    st.sidebar.subheader("Choose outlier removal threshold")
    outlier_threshold = st.sidebar.selectbox("Remove % of most extreme data points for each variable", np.array([5, 2.5, 1, 0.1, 0]))
    
    #refence text sidebar
       
    st.sidebar.subheader("Reference")
    st.sidebar.markdown("""Data covers all companies in the Dow Jones Composite Average, S&P 500, S&P Midcap 400, S&P Smallcap 600, Nasdaq 100, Nasdaq Composite, Russell 1000, Russell 2000, Russell 3000.<br> In total 3148 companies.<br> Data Source is [Finbox](https://finbox.com/screener) (sourced from S&P Capital IQ).""",unsafe_allow_html=True)
    
    
    if price_multiple == 'EV to EBITDA ratio':
        price_multiple = 'EV_TTM_EBITDA'
        
        #filter data
        df_filtered = df[['Sector', 'EV', 'EBITDA',price_multiple, 
                          'EBITDA_Forecast_CAGR_10y', 
                          'Avg_EBITDA_Margin_Forecast_10y', 
                          'Beta_5y', 'Net_Debt_perc_EV']]
        
        df_filtered = df_filtered[df_filtered['Sector']!='Financials']
        df_filtered = df_filtered[df_filtered['EV']>0]
        df_filtered = df_filtered[df_filtered['EBITDA']>0]
        
        if sector != 'All':
            df_filtered = df_filtered[df_filtered['Sector']==sector]
        
        #drop na
        df_filtered = df_filtered.dropna()
        
        #filter for numeric
        df_filtered_num = df_filtered[[price_multiple, 'EBITDA_Forecast_CAGR_10y', 
                          'Avg_EBITDA_Margin_Forecast_10y', 'Beta_5y', 'Net_Debt_perc_EV']]
        
        #remove outliers
        anomaly_matrix = np.zeros(df_filtered_num.shape[0]*df_filtered_num.shape[1])
        anomaly_matrix= anomaly_matrix.reshape(df_filtered_num.shape[0], df_filtered_num.shape[1])
        anomaly_matrix = pd.DataFrame(anomaly_matrix)
        anomaly_matrix.columns = df_filtered_num.columns
             
                
        for variable in df_filtered_num.columns:
            gm = GaussianMixture(n_components=1)
            gm.fit(df_filtered_num[[variable]],)
            densities = gm.score_samples(df_filtered_num[[variable]])
            threshold = np.percentile(densities, outlier_threshold)
            anomaly_matrix[[variable]] = densities < threshold
           
        anomaly_matrix = np.array([np.sum(anomaly_matrix, axis=1)])
        anomaly_matrix = anomaly_matrix.reshape(-1, 1)

        df_filtered = df_filtered[anomaly_matrix==0]
        
        #plot data
        
        #1. EV to EBITD ratio
        st.header("1. EV to EBITDA ratio")
        
        #histogram
        st.subheader('Histogram')
        fig, ax = plt.subplots()
        sns.histplot(data=df_filtered, x=price_multiple, kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered[price_multiple].describe())

        #correlation matrix
        st.subheader('Correlation Matrix of EV to EBITDA ratio vs. Financial KPIs')
        corr = df_filtered[[price_multiple, 'EBITDA_Forecast_CAGR_10y', 
                          'Avg_EBITDA_Margin_Forecast_10y', 'Beta_5y', 'Net_Debt_perc_EV']].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))         # Generate a mask for the upper triangle
        f, ax = plt.subplots()
        cmap = sns.diverging_palette(230, 20, as_cmap=True) # Generate a custom diverging colormap
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        st.pyplot(f)
        
        #2. Expected CAGR 10 EBITDA growth
        st.header("2. Forecasted EBITDA growth next 10 years")
        
        #histogram
        st.subheader('Histogram')
        fig, ax = plt.subplots()
        sns.histplot(data=df_filtered, x='EBITDA_Forecast_CAGR_10y', kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered['EBITDA_Forecast_CAGR_10y'].describe())
        
        #EV to EBITDA vs. expected growth
        st.subheader("EV to EBITDA vs. Forecasted EBITDA growth next 10 years")
        g2 = sns.jointplot(x="EBITDA_Forecast_CAGR_10y", y=price_multiple, data = df_filtered,
                  kind="reg", truncate=False,)
        sns.despine(left=True, bottom=True)
        st.pyplot(g2)
        
        #3. Expected average EBITDA margin next 10 years
        st.header("3. Forecasted EBITDA Margin next 10 years")

        #histogram
        st.subheader('Histogram')
        fig, ax = plt.subplots()
        ax = sns.histplot(data=df_filtered, x='Avg_EBITDA_Margin_Forecast_10y', kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered['Avg_EBITDA_Margin_Forecast_10y'].describe())
        
        #EV to EBITDA vs. expected EBITDA margin
        st.subheader("EV to EBITDA vs. forecasted EBITDA Margin next 10 years")
        g3 = sns.jointplot(x="Avg_EBITDA_Margin_Forecast_10y", y=price_multiple, data = df_filtered,
                  kind="reg", truncate=False,)
        sns.despine(left=True, bottom=True)
        st.pyplot(g3)
        
        #4. Beta past 5 years
        st.header("4. Beta past 5 years")

        #Histrogram
        st.subheader("Histrogram")
        fig, ax = plt.subplots()
        ax = sns.histplot(data=df_filtered, x='Beta_5y', kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered['Beta_5y'].describe())
        
        #EV to EBITDA vs. Beta
        st.subheader("EV to EBITDA vs. Beta past 5 years")
        g4 = sns.jointplot(x="Beta_5y", y=price_multiple, data = df_filtered,
                  kind="reg", truncate=False,)
        sns.despine(left=True, bottom=True)
        st.pyplot(g4)
        
        #5. Net Debt % Enterprise Value
        st.header("5. Net Debt % EV")
        
        #Histrogram
        st.subheader("Histrogram")
        fig, ax = plt.subplots()
        ax = sns.histplot(data=df_filtered, x="Net_Debt_perc_EV", kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered["Net_Debt_perc_EV"].describe())
        
        #EV to EBITDA vs. Net debt % EV
        st.subheader("EV_TTM_EBITDA multiple vs. Net Debt as % of Enterprise Value")
        g5 = sns.jointplot(x="Net_Debt_perc_EV", y=price_multiple, data = df_filtered,
                  kind="reg", truncate=False)
        sns.despine(left=True, bottom=True)
        st.pyplot(g5)
        
      
    
    if price_multiple == 'Price to Earnings ratio':
        price_multiple = 'Market_Cap_TTM_Net_Income'
        
        #filter data
        df_filtered = df[['Sector', 'Market_Cap', 'Net_Income', price_multiple, 
                          'Net_Income_Forecast_CAGR_10y', 
                          'Avg_Net_Income_Margin_Forecast_10y', 
                          'Beta_5y', 'Net_Debt_perc_EV']]
        
        df_filtered = df_filtered[df_filtered['Market_Cap']>0]
        df_filtered = df_filtered[df_filtered['Net_Income']>0]
        
        if sector != 'All':
            df_filtered = df_filtered[df_filtered['Sector']==sector]

        #drop na
        df_filtered = df_filtered.dropna()
    
        #filter for numeric
        df_filtered_num = df_filtered[[price_multiple, 
                          'Net_Income_Forecast_CAGR_10y', 
                          'Avg_Net_Income_Margin_Forecast_10y', 
                          'Beta_5y', 'Net_Debt_perc_EV']]
        
        #remove outliers
        anomaly_matrix = np.zeros(df_filtered_num.shape[0]*df_filtered_num.shape[1])
        anomaly_matrix= anomaly_matrix.reshape(df_filtered_num.shape[0], df_filtered_num.shape[1])
        anomaly_matrix = pd.DataFrame(anomaly_matrix)
        anomaly_matrix.columns = df_filtered_num.columns
             
                
        for variable in df_filtered_num.columns:
            gm = GaussianMixture(n_components=1)
            gm.fit(df_filtered_num[[variable]],)
            densities = gm.score_samples(df_filtered_num[[variable]])
            threshold = np.percentile(densities, outlier_threshold)
            anomaly_matrix[[variable]] = densities < threshold
           
        anomaly_matrix = np.array([np.sum(anomaly_matrix, axis=1)])
        anomaly_matrix = anomaly_matrix.reshape(-1, 1)

        df_filtered = df_filtered[anomaly_matrix==0]
        
        #plot data

        #1. Price to Earnings ratio
        st.header("1. Price to Earnings ratio")
        
        #histogram
        st.subheader('Histogram')
        fig, ax = plt.subplots()
        sns.histplot(data=df_filtered, x=price_multiple, kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered[price_multiple].describe())

        #correlation matrix
        st.subheader('Correlation Matrix of Price to Earnings ratio vs. Financial KPIs')
        corr = df_filtered[[price_multiple, 
                          'Net_Income_Forecast_CAGR_10y', 
                          'Avg_Net_Income_Margin_Forecast_10y', 
                          'Beta_5y', 'Net_Debt_perc_EV']].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))         # Generate a mask for the upper triangle
        f, ax = plt.subplots()
        cmap = sns.diverging_palette(230, 20, as_cmap=True) # Generate a custom diverging colormap
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        st.pyplot(f)
        
        #2. Expected CAGR 10 Net Income growth
        st.header("2. Forecasted Net Income growth next 10 years")
        
        #histogram
        st.subheader('Histogram')
        fig, ax = plt.subplots()
        sns.histplot(data=df_filtered, x='Net_Income_Forecast_CAGR_10y', kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered['Net_Income_Forecast_CAGR_10y'].describe())
        
        #Price to Earnings vs. expected growth
        st.subheader("Price to Earnings vs. Forecasted Net Income growth next 10 years")
        g2 = sns.jointplot(x="Net_Income_Forecast_CAGR_10y", y=price_multiple, data = df_filtered,
                  kind="reg", truncate=False,)
        sns.despine(left=True, bottom=True)
        st.pyplot(g2)
        
        #3. Expected average Net Income margin next 10 years
        st.header("3. Forecasted Net Income Margin next 10 years")

        #histogram
        st.subheader('Histogram')
        fig, ax = plt.subplots()
        ax = sns.histplot(data=df_filtered, x='Avg_Net_Income_Margin_Forecast_10y', kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered['Avg_Net_Income_Margin_Forecast_10y'].describe())
        
        #Price to Earnings vs. expected Net Income margin
        st.subheader("Price to Earnings vs. forecasted Net Income Margin next 10 years")
        g3 = sns.jointplot(x="Avg_Net_Income_Margin_Forecast_10y", y=price_multiple, data = df_filtered,
                  kind="reg", truncate=False,)
        sns.despine(left=True, bottom=True)
        st.pyplot(g3)
        
        #4. Beta past 5 years
        st.header("4. Beta past 5 years")

        #Histrogram
        st.subheader("Histrogram")
        fig, ax = plt.subplots()
        ax = sns.histplot(data=df_filtered, x='Beta_5y', kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered['Beta_5y'].describe())
        
        #Price to Earnings vs. Beta
        st.subheader("Price to Earnings vs. Beta past 5 years")
        g4 = sns.jointplot(x="Beta_5y", y=price_multiple, data = df_filtered,
                  kind="reg", truncate=False,)
        sns.despine(left=True, bottom=True)
        st.pyplot(g4)
        
        #5. Net Debt % Enterprise Value
        st.header("5. Net Debt % EV")
        
        #Histrogram
        st.subheader("Histrogram")
        fig, ax = plt.subplots()
        ax = sns.histplot(data=df_filtered, x="Net_Debt_perc_EV", kde=True)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
        
        #summary statistics
        st.subheader('Summary Statistics')
        st.table(df_filtered["Net_Debt_perc_EV"].describe())
        
        #Price to Earnings vs. Net debt % EV
        st.subheader("Price to Earnings vs. Net Debt as % of Enterprise Value")
        g5 = sns.jointplot(x="Net_Debt_perc_EV", y=price_multiple, data = df_filtered,
                  kind="reg", truncate=False)
        sns.despine(left=True, bottom=True)
        st.pyplot(g5)


if __name__ == '__main__':
    main()