import json
from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import pandas as pd

import os
import urllib.request

MODEL_FILE = "ICT336_G123_Model.joblib"

MODEL_URL = (
    "https://huggingface.co/spaces/"
    "kaungnayne/hotelprediction/resolve/main/"
    "ICT336_G123_Model.joblib"
)

if not os.path.exists(MODEL_FILE):
    print("Downloading trained model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Model downloaded successfully!")


ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / 'ICT336_G123_Model.joblib')  # Load only trusted files.
INFO = json.loads((ROOT / 'ICT336_G123_model_info.json').read_text(encoding='utf-8'))
FEATURES = INFO['features']

# Read actual training categories from the fitted OneHotEncoder, not guessed options.
preprocessor = MODEL.named_steps['preprocessor']
cat_columns = list(preprocessor.transformers_[1][2])
encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
CATEGORIES = {column: [str(v) for v in values] for column, values in zip(cat_columns, encoder.categories_)}

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


def category(name):
    choices = CATEGORIES[name]
    default = choices[0] if choices else None
    return gr.Dropdown(choices=choices, value=default, label=name.replace('_', ' ').title())


def number(name, default=0, minimum=0, maximum=None, precision=0):
    return gr.Number(value=default, label=name.replace('_', ' ').title(), minimum=minimum, maximum=maximum, precision=precision)


def predict(*values):
    booking = dict(zip(FEATURES, values))
    # Validate all required fields and avoid accidental incomplete predictions.
    if any(value is None for value in values):
        raise gr.Error('Please complete all booking fields.')
    if booking['adults'] + booking['children'] + booking['babies'] < 1:
        raise gr.Error('A booking must have at least one guest.')
    if booking['stays_in_weekend_nights'] + booking['stays_in_week_nights'] < 1:
        raise gr.Error('The stay must include at least one night.')
    for key in ['lead_time','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies',
                'previous_cancellations','previous_bookings_not_canceled']:
        if booking[key] < 0:
            raise gr.Error(f'{key} cannot be negative.')
    frame = pd.DataFrame([booking], columns=FEATURES)
    probability = float(MODEL.predict_proba(frame)[0, list(MODEL.classes_).index(1)])
    prediction = int(MODEL.predict(frame)[0])
    label = 'Cancellation predicted' if prediction == 1 else 'No cancellation predicted'
    message = (f'### {label}\n**Estimated cancellation probability: {probability:.1%}**\n\n'
               'This is a statistical estimate, not a guarantee or an automatic booking decision.')
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh(['Not cancelled', 'Cancelled'], [1-probability, probability], color=['#2176ae','#e68a42'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Predicted probability')
    for i, v in enumerate([1-probability, probability]):
        ax.text(min(v + 0.015, 0.95), i, f'{v:.1%}', va='center')
    fig.tight_layout()
    return message, fig

with gr.Blocks(title='ICT336 G123 | Hotel Cancellation Prediction') as demo:
    gr.Markdown('# Hotel Booking Cancellation Prediction\n**ICT336 · Group 123 · Random Forest**')
    gr.Markdown('Enter reservation details to estimate whether a booking may be cancelled. No booking inputs are intentionally saved by this application.')
    inputs = {}
    with gr.Tab('Predict cancellation'):
        with gr.Accordion('Hotel and arrival details', open=True):
            with gr.Row():
                inputs['hotel'] = category('hotel')
                inputs['lead_time'] = number('lead_time', 30)
                inputs['arrival_date_year'] = number('arrival_date_year', 2017, minimum=2015, maximum=2100)
            with gr.Row():
                inputs['arrival_date_month'] = category('arrival_date_month')
                inputs['arrival_date_week_number'] = number('arrival_date_week_number', 25, minimum=1, maximum=53)
                inputs['arrival_date_day_of_month'] = number('arrival_date_day_of_month', 15, minimum=1, maximum=31)
        with gr.Accordion('Stay and guests', open=True):
            with gr.Row():
                inputs['stays_in_weekend_nights'] = number('stays_in_weekend_nights', 1)
                inputs['stays_in_week_nights'] = number('stays_in_week_nights', 2)
                inputs['adults'] = number('adults', 2)
            with gr.Row():
                inputs['children'] = number('children', 0)
                inputs['babies'] = number('babies', 0)
                inputs['meal'] = category('meal')
        with gr.Accordion('Booking details', open=False):
            with gr.Row():
                inputs['market_segment'] = category('market_segment')
                inputs['distribution_channel'] = category('distribution_channel')
                inputs['reserved_room_type'] = category('reserved_room_type')
            with gr.Row():
                inputs['deposit_type'] = category('deposit_type')
                inputs['customer_type'] = category('customer_type')
        with gr.Accordion('Guest booking history', open=False):
            with gr.Row():
                inputs['is_repeated_guest'] = gr.Dropdown(choices=[0, 1], value=0, label='Repeated guest? (1 = Yes)')
                inputs['previous_cancellations'] = number('previous_cancellations', 0)
                inputs['previous_bookings_not_canceled'] = number('previous_bookings_not_canceled', 0)
        predict_button = gr.Button('Predict cancellation risk', variant='primary')
        result = gr.Markdown()
        chart = gr.Plot(label='Cancellation probability chart')
        predict_button.click(predict, inputs=[inputs[f] for f in FEATURES], outputs=[result, chart])
    with gr.Tab('Model performance'):
        gr.Markdown(f'''### Random Forest — revised hold-out evaluation
| Metric | Score |
|---|---:|
| Accuracy | {INFO['test_accuracy']:.2%} |
| Precision | {INFO['test_precision']:.2%} |
| Recall | {INFO['test_recall']:.2%} |
| F1-score | {INFO['test_f1']:.2%} |
| 5-fold cross-validation F1 | {INFO['cv_f1']:.2%} |

**Provisional F1 target:** 75% — **not reached** in the revised evaluation.

**Hotel-type performance (revised hold-out):** City Hotel F1 = 0.7684, recall = 0.7996; Resort Hotel F1 = 0.6713, recall = 0.6909. F1 gap = 0.0971.
''')
    with gr.Tab('About and limitations'):
        gr.Markdown('''### About this project
This educational project predicts hotel booking cancellation using a supervised Random Forest classifier and 20 booking-related features. Dataset: **Hotel Booking Demand**, Antonio, Almeida and Nunes (2019), distributed via the TidyTuesday hotels dataset.

**Source:** https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/hotels.csv

**Limitations:** Predictions are estimates, not guarantees. The dataset describes historical bookings and may not reflect current travel patterns. Performance differs between City and Resort hotels. Some features may change after initial reservation; the deployment assumes the entered values were available at the chosen prediction time. The revised hold-out was created after earlier test results had been examined, so it is **not fully independent validation**. Dates and booking details outside the training period may produce unreliable results. Do not use predictions as the sole basis for denying or modifying a reservation.

**Privacy:** This app does not implement input logging or a database. Hosting providers may still retain standard operational logs. Do not enter real customer names or personal identifiers.
''')

if __name__ == '__main__':
    
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)

