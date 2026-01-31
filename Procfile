release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
web: gunicorn config.wsgi:application --workers 1 --threads 2 --timeout 60 --log-file -
