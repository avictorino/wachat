release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
web: gunicorn config.wsgi:application --workers 2 --threads 4 --timeout 60 --log-file -
