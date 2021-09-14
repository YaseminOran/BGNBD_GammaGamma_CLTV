from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler



def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


creds = {'user': '.............',
         'passwd': '.............',
         'host': '...........,
         'port':...........,
         'db': '..........'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))
pd.read_sql_query("select * from online_retail_2009_2010 limit 10", conn)

retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)
retail_mysql_df.shape
retail_mysql_df.info()
df = retail_mysql_df.copy()

# Veri Ön İşleme
df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T


df_UK = df[df["Country"] == "United Kingdom"]
df_UK.head()
df_UK["TotalPrice"] = df_UK["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

# Lifetime United Kingdom Veri Yapısının Hazırlanması
cltv_df = df_UK.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df.head()
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df = cltv_df[cltv_df["monetary"] > 0]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

cltv_df["frequency"] = cltv_df["frequency"].astype(int)



#BG-NBD Modelinin Kurulması
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

cltv_df.head()

# 1 haftalık  United Kingdom  en iyi 10 müşteri

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
cltv_df.sort_values("expected_purc_1_week", ascending=False).head(20)


#1 ay içinde en çok satın alma beklediğimiz 10 müşteri
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


cltv_df.sort_values("expected_purc_1_month", ascending=False).head(20)








#GAMMA-GAMMA Modelinin Kurulması
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])


ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

# en karlı ilk 10 müşteri
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(20)


##BG-NBD ve GG modeli ile CLTV'nin hesaplanması.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)


cltv = cltv.reset_index()
cltv.sort_values(by= "clv", ascending=False).head()

# 6 aylık zaman periyondunda en değerli 50 müşteri
cltv.sort_values(by="clv", ascending=False).head(50)

#1 aylık ile 12 aylık kıyaslaması

cltv_month = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=1,  # 1 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

cltv_month = cltv_month.reset_index()
cltv_month.head()

cltv_year = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=12,  # 12 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

cltv_year.head()
cltv_year = cltv_year.reset_index()
cltv_month.sort_values(by="clv", ascending=False).head(50)
cltv_year.sort_values(by="clv", ascending=False).head(50)



cltv_month = pd.DataFrame(cltv_month)
cltv_year = pd.DataFrame(cltv_year)

cltv_m_y = cltv_month.merge(cltv_year, on="CustomerID", how="left")
cltv_m_y.head()
cltv_m_y = cltv_m_y.reset_index()



cltv_m_y.drop("index", axis=1, inplace=True)
#cltv_m_y.drop("index_x", axis=1, inplace=True)
#cltv_m_y.drop("index_y", axis=1, inplace=True)

cltv_m_y.columns = ["Customer ID", "1_month_clv", "12_month_clv"]
cltv_m_y.head(10).sort_values("12_month_clv", ascending= False)



cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)


scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])


cltv_final.sort_values(by="scaled_clv", ascending=False).head()


## CLTV'ye Göre Segmentlerin Oluşturulması

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head()


cltv_final.sort_values(by="scaled_clv", ascending=False).head(50)


cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


#veriyi yollama
pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)


cltv_final.head()

cltv_final["Customer ID"] = cltv_final["CustomerID"].astype(int)

cltv_final.to_sql(name='Yasemin_Arslan', con=conn, if_exists='replace', index=False)



pd.read_sql_query("select * from Yasemin_Arslan limit 10", conn)
# conn.close()

