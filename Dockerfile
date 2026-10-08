# Wine 90+ classification: trains the models, generates all results and serves the dashboard.
#
#   docker build -t wine-dashboard .
#   docker run --rm -p 8000:8000 wine-dashboard
#   -> open http://localhost:8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY wine_classification.py export_interactive_data.py ./
COPY ["Full Wine Data.xlsx", "./"]
COPY dashboard/index.html dashboard/app.js dashboard/

# Train and evaluate the models: writes results/ and the dashboard data files.
# export_interactive_data.py also verifies its predictions against results/predictions/.
RUN python wine_classification.py && python export_interactive_data.py

EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--bind", "0.0.0.0", "--directory", "/app/dashboard"]
