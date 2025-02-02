import base64
import pickle

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html
from tensorflow.keras.models import load_model



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

# Load Keras models
model_sec = load_model('pred_sport_from_type.keras')
model_main = load_model('pred_sport_from_all1.keras')

# Load Sports Category Data
gender_sort = pd.read_csv("Sports_Cat.csv")
# Load necessary files and models
with open('one_hot_encoder_sec.pkl', 'rb') as f:
    one_hot_encoder_sec_func = pickle.load(f)

with open('scaler_sec.pkl', 'rb') as f:
    scaler_sec_func = pickle.load(f)

with open('ordinal_sec.pkl', 'rb') as f:
    ordinal_sec_func = pickle.load(f)

df = pd.read_csv("olympics_cleaned.csv")

model_sec = load_model('pred_sport_from_type.keras')

# Define options for dropdown menus
age_options = [{'label': str(i), 'value': i} for i in range(15, 61)]
weight_options = [{'label': str(i), 'value': i} for i in range(25, 201)]
height_options = [{'label': str(i), 'value': i} for i in range(150, 251)]
gender_options = [{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}]
country_options = [{'label': country, 'value': country} for country in [
    "China", "Denmark", "Netherlands", "USA", "Finland", "Norway", "Romania", "Estonia", "France", "Morocco",
    "Spain", "Egypt", "Iran", "Bulgaria", "Italy", "Azerbaijan", "Sudan", "Russia", "Argentina", "Cuba", "Belarus",
    "Greece", "Cameroon", "Turkey", "Chile", "Mexico", "Nicaragua", "Hungary", "Nigeria", "Chad", "Algeria", "Kuwait",
    "Bahrain", "Pakistan", "Iraq", "Syria", "Lebanon", "Qatar", "Malaysia", "Germany", "Canada", "Ireland", "Australia",
    "South Africa", "Eritrea", "Tanzania", "Jordan", "Tunisia", "Libya", "Belgium", "Djibouti", "Palestine", "Comoros",
    "Kazakhstan", "Brunei", "India", "Saudi Arabia", "Maldives", "Ethiopia", "United Arab Emirates", "Yemen", "Indonesia",
    "Philippines", "Uzbekistan", "Kyrgyzstan", "Tajikistan", "Japan", "Switzerland", "Brazil", "Monaco", "Israel", "Sweden",
    "Virgin Islands, US", "Sri Lanka", "Armenia", "Ivory Coast", "Kenya", "Benin", "Ukraine", "UK", "Ghana", "Somalia",
    "Latvia", "Niger", "Mali", "Poland", "Costa Rica", "Panama", "Georgia", "Slovenia", "Croatia", "Guyana", "New Zealand",
    "Portugal", "Paraguay", "Angola", "Venezuela", "Colombia", "Bangladesh", "Peru", "Uruguay", "Puerto Rico", "Uganda",
    "Honduras", "Ecuador", "El Salvador", "Turkmenistan", "Mauritius", "Seychelles", "Czech Republic", "Luxembourg",
    "Mauritania", "Saint Kitts", "Trinidad", "Dominican Republic", "Saint Vincent", "Jamaica", "Liberia", "Suriname",
    "Nepal", "Mongolia", "Austria", "Palau", "Lithuania", "Togo", "Namibia", "Curacao", "Iceland", "American Samoa",
    "Samoa", "Rwanda", "Dominica", "Haiti", "Malta", "Cyprus", "Guinea", "Belize", "South Korea", "Bermuda", "Serbia",
    "Sierra Leone", "Papua New Guinea", "Afghanistan", "Individual Olympic Athletes", "Oman", "Fiji", "Vanuatu", "Moldova",
    "Bahamas", "Guatemala", "Virgin Islands, British", "Mozambique", "Central African Republic", "Madagascar",
    "Bosnia and Herzegovina", "Guam", "Cayman Islands", "Slovakia", "Barbados", "Guinea-Bissau", "Thailand", "Timor-Leste",
    "Democratic Republic of the Congo", "Gabon", "San Marino", "Laos", "Botswana", "North Korea", "Senegal", "Cape Verde",
    "Equatorial Guinea", "Boliva", "Andorra", "Antigua", "Zimbabwe", "Grenada", "Saint Lucia", "Micronesia", "Myanmar",
    "Malawi", "Zambia", "Taiwan", "Sao Tome and Principe", "Republic of Congo", "Macedonia", "Tonga", "Liechtenstein",
    "Montenegro", "Gambia", "Solomon Islands", "Cook Islands", "Albania", "Swaziland", "Burkina Faso", "Burundi", "Aruba",
    "Nauru", "Vietnam", "Cambodia", "Bhutan", "Marshall Islands", "Kiribati", "Kosovo", "South Sudan", "Lesotho"
]]
sport_type_options = [{'label': sport_type, 'value': sport_type} for sport_type in [
    "TeamSports", "CombatSports", "WinterSports", "Athletics", "Aquatics", "RacquetSports", "WaterSports",
    "IndividualSports", "Weightlifting", "Equestrianism", "Shooting", "Cycling", "ModernPentathlon", "Archery", "Triathlon"
]]

def image_source(img):
    image_filename = f'assets/{img}'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    src='data:image/png;base64,{}'.format(encoded_image.decode())
    return src

def create_layout():
    navbar = dbc.Navbar(
        [
            dbc.NavbarBrand(
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(src=image_source("olympics.png"), height="60px", style={"margin-left": "70px"}),
                                style={"margin-right": "5px"},
                            ),
                            dbc.Col(html.H1("Predictions", className="ml-2 align-self-center", style={"font-size": "20px", "text-decoration": "none", "color": "#2D3C6B", "text-decoration": "none", "font-weight": "bold"})),
                        ],
                        align="center",
                    ),
                    href="/",
                    style={"text-decoration": "none"}
                )
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink(html.Img(src=image_source("home.png"), height="40px"), href="/", style={"padding-left": "20px"})),
                    dbc.NavItem(dbc.NavLink(html.Img(src=image_source("pred.png"), height="40px"), href="/", style={"padding-left": "20px"})),
                ],
                className="mr-auto",
                navbar=True,
                style = {
                    "padding-left": "950px"
                }
            ),
        ],
        color="light",
        dark=False,
        className="shadow-sm mb-5 bg-white",
        sticky="top",
        style={
            "height": "80px",
            "width": "100%", 
        }
    )

    layout = dbc.Card(
        html.Div([
            html.Div([
                navbar
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Please Enter Your Data", className='display-5', style={
                            'margin-bottom': '40px',
                            'font_weight': 'bold',
                            'color': 'black',
                            'margin-top': '10px',
                            'text-align': 'left',
                            'font-size': '20px'
                        }),
                        html.Label('Age'),
                        dcc.Dropdown(id='age-dropdown', options=age_options, value=26,
                                    style={
                                        'position': 'relative',
                                        'width': '100%',
                                        'margin-right': '1rem',
                                        'appearance': 'none',
                                        'cursor': 'pointer',
                                        'padding': '3px',
                                        'border-radius': '5px',
                                        'background-color': '#F8F9FA',  
                                        'list-style': 'none',
                                        'height': 'auto' 
                                    }),
                        html.Label('Weight'),
                        dcc.Dropdown(id='weight-dropdown', options=weight_options, value=66,
                                    style={
                                        'position': 'relative',
                                        'width': '100%',
                                        'margin-right': '1rem',
                                        'appearance': 'none',
                                        'cursor': 'pointer',
                                        'padding': '3px',
                                        'border-radius': '5px',
                                        'background-color': '#F8F9FA',  
                                        'list-style': 'none',
                                        'height': 'auto' 
                                    }),
                        html.Label('Height'),
                        dcc.Dropdown(id='height-dropdown', options=height_options, value=169,
                                    style={
                                        'position': 'relative',
                                        'width': '100%',
                                        'margin-right': '1rem',
                                        'appearance': 'none',
                                        'cursor': 'pointer',
                                        'padding': '3px',
                                        'border-radius': '5px',
                                        'background-color': '#F8F9FA',  
                                        'list-style': 'none',
                                        'height': 'auto' 
                                    }),
                        html.Label('Gender'),
                        dcc.Dropdown(id='gender-dropdown', options=gender_options, value='Male',
                                    style={
                                        'position': 'relative',
                                        'width': '100%',
                                        'margin-right': '1rem',
                                        'appearance': 'none',
                                        'cursor': 'pointer',
                                        'padding': '3px',
                                        'border-radius': '5px',
                                        'background-color': '#F8F9FA',  
                                        'list-style': 'none',
                                        'height': 'auto' 
                                    }),
                        html.Label('Country'),
                        dcc.Dropdown(id='country-dropdown', options=country_options, value='Egypt',
                                    style={
                                        'position': 'relative',
                                        'width': '100%',
                                        'margin-right': '1rem',
                                        'appearance': 'none',
                                        'cursor': 'pointer',
                                        'padding': '3px',
                                        'border-radius': '5px',
                                        'background-color': '#F8F9FA',  
                                        'list-style': 'none',
                                        'height': 'auto' 
                                    }),
                        html.Label('Sport Type'),
                        dcc.Dropdown(id='sport-type-dropdown', options=sport_type_options, value='Athletics',
                                    style={
                                        'position': 'relative',
                                        'width': '100%',
                                        'margin-right': '1rem',
                                        'appearance': 'none',
                                        'cursor': 'pointer',
                                        'padding': '3px',
                                        'border-radius': '5px',
                                        'background-color': '#F8F9FA',  
                                        'list-style': 'none',
                                        'height': 'auto' 
                                    }),
                        dbc.Button('Predict', id='button', className='mt-3',
                                style={
                                    'width': '98%',
                                    # 'background-color': '#4CAF50',
                                    'color': 'white',
                                    'padding': '10px',
                                    'margin': 'auto',
                                    'margin-top': '20px',
                                    'border': 'none',
                                    'border-radius': '4px',
                                    'cursor': 'pointer',
                                }),
                    ], className='bg-light p-3 mb-3', style={'margin-bottom': '20px'}),
            dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Output", className='display-5', style={
                                    'font_weight': 'bold',
                                    'color': 'black',
                                    'text-align': 'left',
                                    'font-size': '20px'
                                }),
                                html.Div(id='output', className='text-right', style={'overflow': 'auto', 'margin-top': '0'})
                            ], className='bg-light p-3', style={'margin-bottom': '20px'})
                        ], md=12)
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Top 5 Athletes in the Sport", className='display-4', style={
                            'margin-bottom': '20px',
                            'text-align': 'left',
                            'font-size': '20px',
                            'color': 'black'
                        }),
                        dbc.CardBody([
                            html.Div(id='medal-counts-table', className='mt-3')
                        ])
                    ], className="shadow p-3 mb-5 bg-gray-100 rounded"),
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id='medal-counts-plot', className='mt-3', style={'width': '100%'})
                        ])
                    ], className="shadow p-3 mb-5 bg-gray-100 rounded")
                ], md=9)
            ])
        ])
    )

    return layout



def create_callbacks(app):
    @app.callback(
        [Output('output', 'children'),
        Output('medal-counts-table', 'children'),
        Output('medal-counts-plot', 'children')],
        [Input('button', 'n_clicks')],
        [Input('age-dropdown', 'value')],
        [Input('weight-dropdown', 'value')],
        [Input('height-dropdown', 'value')],
        [Input('gender-dropdown', 'value')],
        [Input('country-dropdown', 'value')],
        [Input('sport-type-dropdown', 'value')]
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
                style_data={'textAlign': 'center'},
                style_cell={'textAlign': 'center'},  # Center align all cells
        style_cell_conditional=[  # Center align only the 'Name' column
            {
                'if': {'column_id': 'Name'},
                'textAlign': 'center'
            }
        ]
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

            fig.update_layout(
            width=800,  # Set the width to 800 pixels
            title={'text': "Medal Counts by Top 5 Athlete in the sport", 'x': 0.5},
    )

        return html.Div([
                        html.H3("The predicted sport according to the sport type is:", style={'margin-top': '60px' , 'font-size': '20px'}),
                        dbc.Row([
                            dbc.Col([
                                html.Ul([html.Li(f"{out_put.split(' (')[0]}")])
                            ], md=12)
                        ]),
                        html.H3("The predicted medals based on the recommended sports: " , style={'margin-top': '20px' , 'font-size': '20px'}),
                        dbc.Row([
                            dbc.Col([
                                html.Ul([html.Li(f"{item[0]} in {item[1]} of sport type {item[2].split(' (')[0]}") for item in sorted_data])
                            ], md=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Ul("")
                            ], md=12)
                        ])
                    ]), table, dcc.Graph(figure=fig, style={'width': '40%', 'margin-top': '10px', 'margin-left': '80px'})


