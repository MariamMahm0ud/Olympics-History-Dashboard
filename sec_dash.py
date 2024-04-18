import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from dash import dash_table
import plotly.express as px

# Load transformers and models
with open('one_hot_encoder_sec.pkl', 'rb') as f:
    one_hot_encoder_sec_func = pickle.load(f)

with open('scaler_sec.pkl', 'rb') as f:
    scaler_sec_func = pickle.load(f)

with open('ordinal_sec.pkl', 'rb') as f:
    ordinal_sec_func = pickle.load(f)

with open('one_hot_encoder_main.pkl', 'rb') as f:
    one_hot_encoder_main_func = pickle.load(f)

with open('scaler_main.pkl', 'rb') as f:
    scaler_main_func = pickle.load(f)

with open('features_main.pkl', 'rb') as f:
    features_main_func = pickle.load(f)

# Load Sports Category Data
gender_sort = pd.read_csv("Sports_Cat.csv")

df = pd.read_csv("olympics_cleaned.csv")

model_main=load_model('pred_sport_from_all1.keras')

model_sec = load_model('pred_sport_from_type.keras')

# Define options for dropdown menus
age_options = [{'label': str(i), 'value': i} for i in range(15, 61)]
weight_options = [{'label': str(i), 'value': i} for i in range(25, 201)]
height_options = [{'label': str(i), 'value': i} for i in range(150, 251)]
gender_options = [{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}]
country_options = [{'label': country, 'value': country} for country in ["China", "Denmark", "Netherlands", "USA", "Finland", "Norway", "Romania", "Estonia", "France", "Morocco", "Spain", "Egypt", "Iran", "Bulgaria", "Italy", "Azerbaijan", "Sudan", "Russia", "Argentina", "Cuba", "Belarus", "Greece", "Cameroon", "Turkey", "Chile", "Mexico", "Nicaragua", "Hungary", "Nigeria", "Chad", "Algeria", "Kuwait", "Bahrain", "Pakistan", "Iraq", "Syria", "Lebanon", "Qatar", "Malaysia", "Germany", "Canada", "Ireland", "Australia", "South Africa", "Eritrea", "Tanzania", "Jordan", "Tunisia", "Libya", "Belgium", "Djibouti", "Palestine", "Comoros", "Kazakhstan", "Brunei", "India", "Saudi Arabia", "Maldives", "Ethiopia", "United Arab Emirates", "Yemen", "Indonesia", "Philippines", "Uzbekistan", "Kyrgyzstan", "Tajikistan", "Japan", "Switzerland", "Brazil", "Monaco", "Israel", "Sweden", "Virgin Islands, US", "Sri Lanka", "Armenia", "Ivory Coast", "Kenya", "Benin", "Ukraine", "UK", "Ghana", "Somalia", "Latvia", "Niger", "Mali", "Poland", "Costa Rica", "Panama", "Georgia", "Slovenia", "Croatia", "Guyana", "New Zealand", "Portugal", "Paraguay", "Angola", "Venezuela", "Colombia", "Bangladesh", "Peru", "Uruguay", "Puerto Rico", "Uganda", "Honduras", "Ecuador", "El Salvador", "Turkmenistan", "Mauritius", "Seychelles", "Czech Republic", "Luxembourg", "Mauritania", "Saint Kitts", "Trinidad", "Dominican Republic", "Saint Vincent", "Jamaica", "Liberia", "Suriname", "Nepal", "Mongolia", "Austria", "Palau", "Lithuania", "Togo", "Namibia", "Curacao", "Iceland", "American Samoa", "Samoa", "Rwanda", "Dominica", "Haiti", "Malta", "Cyprus", "Guinea", "Belize", "South Korea", "Bermuda", "Serbia", "Sierra Leone", "Papua New Guinea", "Afghanistan", "Individual Olympic Athletes", "Oman", "Fiji", "Vanuatu", "Moldova", "Bahamas", "Guatemala", "Virgin Islands, British", "Mozambique", "Central African Republic", "Madagascar", "Bosnia and Herzegovina", "Guam", "Cayman Islands", "Slovakia", "Barbados", "Guinea-Bissau", "Thailand", "Timor-Leste", "Democratic Republic of the Congo", "Gabon", "San Marino", "Laos", "Botswana", "North Korea", "Senegal", "Cape Verde", "Equatorial Guinea", "Boliva", "Andorra", "Antigua", "Zimbabwe", "Grenada", "Saint Lucia", "Micronesia", "Myanmar", "Malawi", "Zambia", "Taiwan", "Sao Tome and Principe", "Republic of Congo", "Macedonia", "Tonga", "Liechtenstein", "Montenegro", "Gambia", "Solomon Islands", "Cook Islands", "Albania", "Swaziland", "Burkina Faso", "Burundi", "Aruba", "Nauru", "Vietnam", "Cambodia", "Bhutan", "Marshall Islands", "Kiribati", "Kosovo", "South Sudan", "Lesotho"]]
sport_type_options = [{'label': sport_type, 'value': sport_type} for sport_type in ["TeamSports", "CombatSports", "WinterSports", "Athletics", "Aquatics", "RacquetSports", "WaterSports", "IndividualSports", "Weightlifting", "Equestrianism", "Shooting", "Cycling", "ModernPentathlon", "Archery", "Triathlon"]]


def create_layout():
    layout = dbc.Card(
        html.Div([
            html.Div(children=[
                dcc.Link('Home', href='/', className="btn btn-dark m-2 fs-5"),
                dcc.Link('Plot Page', href='/plot', className="btn btn-dark m-2 fs-5")
            ]),
            html.H1("Sport Prediction By Sport Type", className='display-4', style={'margin-bottom': '40px','text-align': 'left','font-weight': 'bold','color': 'black','font-size': '43px', 'border-bottom': '2px solid #000000'}),
            dbc.Row([
                dbc.Col([
                    
                    html.H1("Please enter your body features and the desired Sport type", className='display-5', style={'margin-bottom': '40px','font_weight': 'bold','color': 'black','margin-top': '10px','text-align': 'left', 'font-size': '25px'}),
                    html.Label('Age'),
                    dcc.Dropdown(id='age-dropdown', options=age_options, value=26),
                    html.Label('Weight'),
                    dcc.Dropdown(id='weight-dropdown', options=weight_options, value=66),
                    html.Label('Height'),
                    dcc.Dropdown(id='height-dropdown', options=height_options, value=169),
                    html.Label('Gender'),
                    dcc.Dropdown(id='gender-dropdown', options=gender_options, value='Male'),
                    html.Label('Country'),
                    dcc.Dropdown(id='country-dropdown', options=country_options, value='Egypt'),
                    html.Label('Sport Type'),
                    dcc.Dropdown(id='sport-type-dropdown', options=sport_type_options, value='Athletics'),
                    dbc.Button('Predict', id='button', color='primary', className='mt-3'),
                    html.Div(id='output', className='mt-3 text-right')
                ], md=3, className='bg-light p-3'),
                dbc.Col([
                    html.H1("Top 5 Athletes in the sport", className='display-4', style={'margin-bottom': '20px','margin-left': '90px','text-align': 'left', 'font-size': '30px','font-weight': 'bold','color': 'black'}),
                    html.Div(id='medal-counts-table', className='mt-3'),
                    html.Div(id='medal-counts-plot', className='mt-3'),
                ], md=9, className='bg-light p-3')
            ])
        ] , className="shadow p-3 mb-5 bg-gray-100 rounded h-100")
    )
    return layout

def create_callbacks(app):
    @app.callback(
        [Output('output', 'children'),
         Output('medal-counts-table', 'children'),
         Output('medal-counts-plot', 'children')],
        [Input('button', 'n_clicks')],
        [Input('age-dropdown', 'value'),
         Input('weight-dropdown', 'value'),
         Input('height-dropdown', 'value'),
         Input('gender-dropdown', 'value'),
         Input('country-dropdown', 'value'),
         Input('sport-type-dropdown', 'value')]
    )
    def update_output(n_clicks, age, weight, height, gender, country, sport_type):
        if n_clicks is None:
            return '', None, None
        
        else:
            rec_s = []
            rec_m = []

            if gender == "Male":
                sport_type_c = sport_type + " (M)"
                cat = gender_sort["male"]
            else:
                sport_type_c = sport_type + " (F)"
                cat = gender_sort["female"]

            # Predict sport for each category
            for sport_type in cat:
                x_sec_func = ordinal_sec_func.transform(pd.DataFrame({"Gender": [gender], "Team_origen": [country], "Sport_Type": [sport_type]}))
                x_sec_func = np.concatenate((pd.DataFrame({"Age": [age], "Height": [height], "Weight": [weight]}).values, x_sec_func), axis=1)
                x_sec_inp = scaler_sec_func.transform(x_sec_func)
                pred = model_sec.predict(x_sec_inp)
                original_data_sec = one_hot_encoder_sec_func.inverse_transform(pred)
                rec_s.append([original_data_sec[0][0], sport_type])

            # Predict sport for recommended categories
            for i in rec_s:
                x_main_func = features_main_func.transform(pd.DataFrame({"Sport": [i[0]], "Gender": [gender], "Team_origen": [country], "Sport_Type": [i[1]]}))
                x_main_func = np.concatenate((pd.DataFrame({"Age": [age], "Height": [height], "Weight": [weight]}).values, x_main_func), axis=1)
                x_main_inp = scaler_main_func.transform(x_main_func)
                pred = model_main.predict(x_main_inp)
                original_data_main = one_hot_encoder_main_func.inverse_transform(pred)
                if original_data_main[0][0] == "NoMedal":
                    continue
                rec_m.append([original_data_main[0][0], i[0].split(" (")[0], i[1] ])

            # Sort recommended categories
            def custom_sort(sublist):
                custom_order = {'Gold': 0, 'Silver': 1, 'Bronze': 2}
                return custom_order.get(sublist[0])
            
            sorted_data = sorted(rec_m, key=custom_sort)

            # Predict sport based on the selected sport type
            x_sec_func = ordinal_sec_func.transform(pd.DataFrame({"Gender": [gender], "Team_origen": [country], "Sport_Type": [sport_type_c]}))
            x_sec_func = np.concatenate((pd.DataFrame({"Age": [age], "Height": [height], "Weight": [weight]}).values, x_sec_func), axis=1)
            x_sec_inp = scaler_sec_func.transform(x_sec_func)
            pred = model_sec.predict(x_sec_inp)

            original_data_sec = one_hot_encoder_sec_func.inverse_transform(pred)
            out_put = original_data_sec[0][0]

            # Generate medal counts table
            medal_counts = df.pivot_table(index=['Name', 'Team_origen', 'Sport'], columns='Medal', aggfunc='size', fill_value=0)
            medal_counts.reset_index(inplace=True)
            medal_counts.columns.name = None
            medal_counts["Medal_Count"] = medal_counts["Gold"] + medal_counts["Silver"] + medal_counts["Bronze"]
            medal_counts["Practice_Count"] = medal_counts["Gold"] + medal_counts["Silver"] + medal_counts["Bronze"] + medal_counts["NoMedal"]
            medal_counts["Medal_Point"] = medal_counts["Gold"] * 3 + medal_counts["Silver"] * 2 + medal_counts["Bronze"]
            medal_counts = medal_counts.sort_values(by="Medal_Point", ascending=False)
            medal_counts = medal_counts[medal_counts.Sport == out_put]
            medal_counts = medal_counts.drop(columns=["NoMedal","Sport"]).reset_index(drop=True).head(5)
            table = dash_table.DataTable(
                # Data
                data=medal_counts.to_dict('records'),
                # Columns
                columns=[{'name': col, 'id': col} for col in medal_counts.columns],
                # Styling
                style_table={'overflowX': 'auto', 'width': '80%', 'float': 'middle', 'margin-top': '10px', 'margin-left': '80px'},
            )

            # Generate medal counts plot
            wide_df = medal_counts
            fig = px.bar(wide_df, x="Name", y=["Gold", "Silver", "Bronze"], title="Medal Counts by Top 5 Athlete in the sport")
            fig.update_xaxes(title_text="Athlete Name")
            fig.update_yaxes(title_text="Medal Count")
            color_mapping = {'Gold': 'goldenrod', 'Silver': 'grey', 'Bronze':'saddlebrown'}
            for i, data in enumerate(fig.data):
                medal_type = data.name
                fig.data[i].marker.color = color_mapping[medal_type]


            return html.Div([
                        html.H3("The predicted sport according to the SPORT TYPE is:", style={'margin-top': '60px' , 'font-size': '20px'}),
                        dbc.Row([
                            dbc.Col([
                                html.Ul([html.Li(f"{out_put.split(' (')[0]}")])
                            ], md=12)
                        ]),
                        html.H3("The predicted medals based on the recommended sports: " , style={'margin-top': '20px' , 'font-size': '20px'}),
                        dbc.Row([
                            dbc.Col([
                                html.Ul([html.Li(f"{item[0]} in {item[1]} of SPORT TYPE {item[2].split(' (')[0]}") for item in sorted_data])
                            ], md=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Ul("")
                            ], md=12)
                        ])
                    ]), table, dcc.Graph(figure=fig, style={'width': '40%', 'margin-top': '10px', 'margin-left': '80px'})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = create_layout()
create_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)