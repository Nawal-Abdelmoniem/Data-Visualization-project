import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from dash import html,dcc,Dash,Input,Output,State
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime


pio.templates.default = "plotly_white"

# Define the pastel color palette
pastel_colors = px.colors.qualitative.Pastel

pd.set_option('display.max_rows',500)
pd.set_option('display.max_colwidth',500)

data = pd.read_csv("rfm_data.csv")


segment_product_type = data.groupby(['ProductInformation', 'Location']).size().reset_index(name='type')

products_seg = px.sunburst(segment_product_type, 
                  path=['ProductInformation', 'Location'], 
                  values='type', 
                  color='ProductInformation', 
                  color_discrete_sequence=px.colors.qualitative.Pastel) 
                  

from datetime import datetime

# Convert 'PurchaseDate' to datetime
data["PurchaseDate"] = pd.to_datetime(data["PurchaseDate"])

# Calculate Recency
data['Recency'] = (datetime.now().date() - data['PurchaseDate'].dt.date)
data['Recency'] = pd.to_timedelta(data['Recency'])
data['Recency'] = data['Recency'].dt.days

# Calculate Frequency
frequency_data = data.groupby('CustomerID')['OrderID'].count().reset_index()
frequency_data.rename(columns={'OrderID': 'Frequency'}, inplace=True)
data = data.merge(frequency_data, on='CustomerID', how='left')


# Calculate Monetary Value
monetary_data = data.groupby('CustomerID')['TransactionAmount'].sum().reset_index()
monetary_data.rename(columns={'TransactionAmount': 'MonetaryValue'}, inplace=True)
data = data.merge(monetary_data, on='CustomerID', how='left')


# Define scoring criteria for each RFM value
recency_scores = [5, 4, 3, 2, 1]  # Higher score for lower recency (more recent)
frequency_scores = [1, 2, 3, 4, 5]  # Higher score for higher frequency
monetary_scores = [1, 2, 3, 4, 5]  # Higher score for higher monetary value

# Calculate RFM scores
data['RecencyScore'] = pd.cut(data['Recency'], bins=5, labels=recency_scores)
data['FrequencyScore'] = pd.cut(data['Frequency'], bins=5, labels=frequency_scores)
data['MonetaryScore'] = pd.cut(data['MonetaryValue'], bins=5, labels=monetary_scores)

# Convert RFM scores to numeric type
data['RecencyScore'] = data['RecencyScore'].astype(int)
data['FrequencyScore'] = data['FrequencyScore'].astype(int)
data['MonetaryScore'] = data['MonetaryScore'].astype(int)

# Calculate RFM score by combining the individual scores
data['RFM_Score'] = data['RecencyScore'] + data['FrequencyScore'] + data['MonetaryScore']


# Create RFM segments based on the RFM score
segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
data['Value Segment'] = pd.qcut(data['RFM_Score'], q=3, labels=segment_labels)

# RFM Segment Distribution
segment_counts = data['Value Segment'].value_counts().reset_index()
segment_counts.columns = ['Value Segment', 'Count']

# Create the bar chart with pastel colors
fig_segment_dist = px.bar(segment_counts, x='Value Segment', y='Count',
                          color='Value Segment', color_discrete_sequence=pastel_colors,
                          title='RFM Value Segment Distribution')

# Update the layout
fig2_segment_dist = fig_segment_dist.update_layout(xaxis_title='RFM Value Segment',
                              yaxis_title='Count',
                              showlegend=False)

# Create a new column for RFM Customer Segments
data['RFM Customer Segments'] = ''

# Assign RFM segments based on the RFM score
data.loc[data['RFM_Score'] >= 9, 'RFM Customer Segments'] = 'Champions'
data.loc[(data['RFM_Score'] >= 6) & (data['RFM_Score'] < 9), 'RFM Customer Segments'] = 'Potential Loyalists'
data.loc[(data['RFM_Score'] >= 5) & (data['RFM_Score'] < 6), 'RFM Customer Segments'] = 'At Risk Customers'
data.loc[(data['RFM_Score'] >= 4) & (data['RFM_Score'] < 5), 'RFM Customer Segments'] = "Can't Lose"
data.loc[(data['RFM_Score'] >= 3) & (data['RFM_Score'] < 4), 'RFM Customer Segments'] = "Lost"


segment_product_counts = data.groupby(['Value Segment', 'RFM Customer Segments']).size().reset_index(name='Count')

segment_product_counts = segment_product_counts.sort_values('Count', ascending=False)

fig_pie = px.pie(segment_product_counts, 
             names='Value Segment', 
             values='Count', 
             title='Proportion of Products by Value Segment')

cust_seg= px.sunburst(segment_product_counts, 
                  path=['Value Segment', 'RFM Customer Segments'], 
                  values='Count', 
                  color='Value Segment', 
                  color_discrete_sequence=px.colors.qualitative.Pastel, 
                  title='RFM Customer Segments by Value')

fig_bar = px.bar(segment_product_counts, 
             x='RFM Customer Segments', 
             y='Count', 
             color='Value Segment', 
             barmode='stack', 
             title='RFM Customer Segments by Value (Stacked Bar)')

fig_treemap_segment_product = px.treemap(segment_product_counts,
                                         path=['Value Segment', 'RFM Customer Segments'],
                                         values='Count',
                                         color='Value Segment', color_discrete_sequence=px.colors.qualitative.Pastel,
                                         title='RFM Customer Segments by Value')

# Filter the data to include only the customers in the Champions segment
champions_segment = data[data['RFM Customer Segments'] == 'Champions']

champions_segment_fig = go.Figure()
champions_segment_fig.add_trace(go.Box(y=champions_segment['RecencyScore'], name='Recency'))
champions_segment_fig.add_trace(go.Box(y=champions_segment['FrequencyScore'], name='Frequency'))
champions_segment_fig.add_trace(go.Box(y=champions_segment['MonetaryScore'], name='Monetary'))

champions_segment_fig.update_layout(title='Distribution of RFM Values within Champions Segment',
                  yaxis_title='RFM Value',
                  showlegend=True)


city_to_country = {
    'Tokyo': 'Japan',
    'London': 'United Kingdom',
    'New York': 'United States',
    'Paris': 'France'
}
data['Country'] = data['Location'].map(city_to_country)

country_transactions = data.groupby('Country').size().reset_index(name='TransactionCount')

map_fig = px.choropleth(country_transactions, 
                    locations='Country', 
                    locationmode='country names', 
                    color='TransactionCount',
                    
                    title='Transaction Count by Country')

correlation_matrix = champions_segment[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].corr()

# Visualize the correlation matrix using a heatmap
fig_corr_heatmap = go.Figure(data=go.Heatmap(
                   z=correlation_matrix.values,
                   x=correlation_matrix.columns,
                   y=correlation_matrix.columns,
                   colorscale='blues',
                   colorbar=dict(title='Correlation')))

fig_corr_heatmap.update_layout(title='RFM Values vs Champions Segment')

segment_counts = data['RFM Customer Segments'].value_counts()

# Create a bar chart to compare segment counts
comparison_fig = go.Figure(data=[go.Bar(x=segment_counts.index, y=segment_counts.values,
                            marker=dict(color=pastel_colors))])

# Set the color of the Champions segment as a different color
champions_color = 'rgb(158, 202, 225)'
comparison_fig.update_traces(marker_color=[champions_color if segment == 'Champions' else pastel_colors[i]
                                for i, segment in enumerate(segment_counts.index)],
                  marker_line_color='rgb(8, 48, 107)',
                  marker_line_width=1.5, opacity=0.6)

# Update the layout
comparison_fig.update_layout(title='Comparison of RFM Segments',
                  xaxis_title='RFM Segments',
                  yaxis_title='Number of Customers',
                  showlegend=False,
                  plot_bgcolor='rgba(0,0,0,0)')

# Calculate the average Recency, Frequency, and Monetary scores for each segment
segment_scores = data.groupby('RFM Customer Segments')[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].mean().reset_index()


# Create a grouped bar chart to compare segment scores
fig_2 = go.Figure()

# Add bars for Recency score
fig_2.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['RecencyScore'],
    name='Recency Score',
    marker_color='rgb(158,202,225)'
))

# Add bars for Frequency score
fig_2.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['FrequencyScore'],
    name='Frequency Score',
    marker_color='rgb(94,158,217)'
))

# Add bars for Monetary score
fig_2.add_trace(go.Bar(
    x=segment_scores['RFM Customer Segments'],
    y=segment_scores['MonetaryScore'],
    name='Monetary Score',
    marker_color='rgb(32,102,148)'
))

# Update the layout
imp_fig = fig_2.update_layout(
    title='Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores',
    xaxis_title='RFM Segments',
    yaxis_title='Score',
    barmode='group',
    showlegend=True
)

# Assumptions
discount_rate = 0.1  # Discount rate
repeat_purchase_rate = 0.2  # Repeat purchase rate

# Customer Lifespan
earliest_purchase_date = data['PurchaseDate'].min().date()
current_date = datetime.now().date()
customer_lifespan = (current_date - earliest_purchase_date).days

# Revenue per transaction
revenue_per_transaction = data['TransactionAmount'].mean()

# Purchase frequency
purchase_frequency = data['Frequency'].mean()

# CLV Calculation
clv = (revenue_per_transaction * purchase_frequency * repeat_purchase_rate) / (1 - discount_rate)

# Visualize CLV
fig_clv = go.Figure(go.Indicator(
    mode="number",
    value=clv,
    title="Estimated CLV",
    number={'prefix': "$"}
))

fig_clv.update_layout(title="Estimated Customer Lifetime Value (CLV)")


data['Month'] = data['PurchaseDate'].dt.to_period('M').astype(str)
data['Quarter'] = data['PurchaseDate'].dt.to_period('Q').astype(str)
 

monthly_sales = data.groupby('Month')['TransactionAmount'].sum().reset_index()
fig_monthly_sales = px.line(monthly_sales, x='Month', y='TransactionAmount', title='Monthly Sales Trends')
fig_monthly_sales.update_xaxes(title='Month')
fig_monthly_sales.update_yaxes(title='Total Sales')

product_sales = data.groupby('ProductInformation')['TransactionAmount'].sum().reset_index().sort_values(by='TransactionAmount', ascending=False)

# Identify top-selling products and their contribution to overall revenue
top_selling_products = product_sales.head(10)
total_revenue = data['TransactionAmount'].sum()
top_selling_products['Contribution (%)'] = (top_selling_products['TransactionAmount'] / total_revenue) * 100

# Plot top-selling products and their contribution to revenue
fig_top_selling_products = px.bar(top_selling_products, x='ProductInformation', y='TransactionAmount',
                                   title='Top Selling Products and Their Contribution to Revenue',
                                   labels={'TransactionAmount': 'Revenue', 'ProductInformation': 'Product ID'},
                                   hover_data=['Contribution (%)'],
                                   color='ProductInformation')
fig_top_selling_products.update_layout(yaxis_title='Revenue', xaxis_title='Product ID')



# Group by date and location to get daily sales trends
daily_sales_location = data.groupby(['PurchaseDate', 'Location'])['TransactionAmount'].sum().reset_index()
daily_sales_location['PurchaseDate'] = pd.to_datetime(daily_sales_location['PurchaseDate'])

# Create figure
fig = go.Figure()

# Add traces for each location
for location in daily_sales_location['Location'].unique():
    location_data = daily_sales_location[daily_sales_location['Location'] == location]
    fig.add_trace(
        go.Scatter(x=location_data['PurchaseDate'], y=location_data['TransactionAmount'],
                   mode='lines', name=location)
    )

# Set title and axis labels
fig.update_layout(
    title_text="Daily Sales Trends by Location",
    xaxis_title="Date",
    yaxis_title="Total Sales"
)

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1d",
                     step="day",
                     stepmode="backward"),
                dict(count=7,
                     label="1w",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)


rfm_segment_counts = data['RFM Customer Segments'].value_counts().reset_index()
rfm_segment_counts.columns = ['RFM Customer Segments', 'Count']
 
# Plot the donut chart
fig_donut = px.pie(rfm_segment_counts, values='Count', names='RFM Customer Segments', hole=0.5,
                    title='Distribution of RFM Customer Segments')
 

# Assuming your DataFrame is named 'data'
product_counts = data['ProductInformation'].value_counts()

# Define a color palette
palette = px.colors.qualitative.Pastel

# Create a donut chart
fig_donut = px.pie(names=product_counts.index, 
                    values=product_counts.values, 
                    hole=0.5, 
                    color_discrete_sequence=palette)

# Update layout to make it look like a donut chart
fig_donut.update_traces(textinfo='percent+label', textposition='inside')
fig_donut.update_layout(title='Product Distribution')

unique_customer_count = data['CustomerID'].nunique()

def create_box_plot(segment):
    segment_data = data[data['RFM Customer Segments'] == segment]
    fig = go.Figure()
    fig.add_trace(go.Box(y=segment_data['RecencyScore'], name='Recency'))
    fig.add_trace(go.Box(y=segment_data['FrequencyScore'], name='Frequency'))
    fig.add_trace(go.Box(y=segment_data['MonetaryScore'], name='Monetary'))
    fig.update_layout(title=f'Box Plot for {segment} Segment',
                      xaxis_title='RFM Scores',
                      yaxis_title='Score Value',
                      showlegend=True)
    return fig


app=Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div(style={"background": "#f3fafd"}, children=[
    
    
   #the header first row 
   dbc.Row([
        html.H1("RFM Analysis Dashboard", style={
            "font-size": "30px",
            "font-weight": "bold",
            "font-family": "Tahoma",
            "color": "white",
            "text-align": "center",
            "background": "#406170",
            "width": "100%",
            "padding": "20px",
            "margin-bottom": "20px"})]),
    
    
    
    #second row : with cards 
    dbc.Row([
      dbc.Col(width=2),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H3(f"{sum(data['TransactionAmount']):,.00f}", id='total-TransactionAmount-card'),
                    html.H4("Transactions", className="text-info")]),
                
                style={"color": "white", "background-color": "#406170", "text-align": "center",
                           "border": "2px solid #77ACF1", "border-radius": "10px", "margin-bottom": "20px",
                           "max-width": "210px", "height": "100px"})]), 
     
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H3(f"{len(data['CustomerID'].unique()):,.0f}", id='Total-Customers-card'),
                    html.H4("Customers", className="text-info")]),
                style={"color": "white", "background-color": "#406170", "text-align": "center",
                   "border": "2px solid #77ACF1", "border-radius": "10px", "margin-bottom": "20px",
                  "max-width": "210px", "height": "100px"}),]),  
        
          dbc.Col([
           dbc.Card(
               dbc.CardBody([
                   html.H3(children=f"{len(data['ProductInformation'].unique()):,.0f}", id='ProductInformation-card'),
                    html.H4("Products", className="text-info")]),
               style={"color": "white", "background-color": "#406170", "text-align": "center",
                    "border": "2px solid #77ACF1", "border-radius": "10px", "margin-bottom": "20px",
                   "max-width": "210px", "height": "100px"}),]),
        ]),

    #third_row 
dbc.Row([
       dbc.Col([
         dbc.Card(
            dbc.CardBody([
                html.H6('Recency Score', className="text-info"),
                dcc.Dropdown(
                    id='recency-dropdown',
                    options=[{'label': str(score), 'value': score} for score in range(1, 6)],
                    value=1,
                    style={'width': '100%','margin-bottom': '20px'}),
                html.H6('Frequency Score', className="text-info"),
                dcc.Dropdown(
                    id='frequency-dropdown',
                    options=[{'label': str(score), 'value': score} for score in range(1, 6)],
                    value=1,
                    style={'width': '100%','margin-bottom': '20px'}),
                html.H6('Monetary Score', className="text-info"),
                dcc.Dropdown(
                    id='monetary-dropdown',
                    options=[{'label': str(score), 'value': score} for score in range(1, 6)],
                    value=1,
                    style={'width': '100%','margin-bottom': '20px'}),
                
                html.Div(id='segment-output', style={"color": "white"})]),
            style={"color": "#406170", "background-color": "#406170", "text-align": "center",
                   "border": "2px solid #77ACF1", "border-radius": "10px", "margin-bottom": "20px",
                   "max-width": "230px", "height": "410px"} ),], width=3,style={'padding-left': '30px','marginTop':'10px'}), 
    
    dbc.Col([
        dcc.Graph(
            id='comparison_fig',
            figure=comparison_fig)], width=4, style={'marginTop':'10px','width':'40%'}),
    
    dbc.Col([
        dcc.Graph(
            id='cust_seg',
            figure=cust_seg )], style={'width': '30%', 'margin_right' : '55px','marginTop':'10px'}, width=3),    
                 ]),
    
    #fourth row 
     dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='segment-dropdown',
                options=[{'label': segment, 'value': segment} for segment in data['RFM Customer Segments'].unique()],
                value='Champions',  
                style={'width': '70%' , 'marginTop':'20px'}
               ),
             dcc.Graph(id='box-plot')], width=5,style={'padding-left': '30px'}),
         
         dbc.Col([
             dcc.Graph( 
                 id='imp_fig',
                 figure = imp_fig )], 
                 style={'width': '45%','margin-right': '50px' , 'margin_left' : '10px' , 'marginTop':'55px'})
     ]),
     #fifth row 
    dbc.Row([
        dbc.Col([
            dcc.Graph( 
                id='fig_donut',
                figure=fig_donut),
            ], width=3,style={'width': '30%', 'margin-left': '55px', 'margin-top': '20px'}),
        
        dbc.Col([
            dcc.Graph( 
                 id='map_fig',
                 figure = map_fig)
        ], 
            style={'width': '80%','margin-right': '55px' , 'margin_left' : '55px' ,'marginTop':'20px' })
     ]),
    
        #six_row
        dbc.Row([
            dbc.Col([
                dcc.Graph( 
                    id='Daily Sales Trends by Location',
                    figure=fig,
                    
                ),
            ],style={'width': '70%', 'margin-left': '55px', 'margin-top': '20px'} ,width=7),
            
            dbc.Col([
                dcc.Graph( 
                    id='Correlation Matrix of RFM Values within Champions Segment',
                    figure = fig_corr_heatmap, 
                ),],style={'width': '30%', 'margin-right': '55px', 'margin-top': '20px'}),

        ])

])

@app.callback(
    Output('segment-output', 'children'),
    [Input('recency-dropdown', 'value'),
     Input('frequency-dropdown', 'value'),
     Input('monetary-dropdown', 'value')]
)
def update_segment(recency_score, frequency_score, monetary_score):
    segment = data[(data['RecencyScore'] == recency_score) &
                   (data['FrequencyScore'] == frequency_score) &
                   (data['MonetaryScore'] == monetary_score)]['RFM Customer Segments']
    if not segment.empty:
        return f"The RFM Customer Segment is: {segment.values[0]}"
    else:
        return "No matching segment found"

@app.callback(
Output('box-plot', 'figure'),
[Input('segment-dropdown', 'value')]
)

def update_graph(selected_segment):
    return create_box_plot(selected_segment)

if __name__ == '__main__':
    app.run_server(debug=False) 