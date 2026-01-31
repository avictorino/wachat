# Heroku Deployment Guide

This guide explains how to deploy the WaChat application to Heroku with PostgreSQL database.

## Prerequisites

- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed
- A Heroku account
- Git repository configured

## Deployment Steps

### 1. Create a Heroku App

```bash
heroku create your-app-name
```

Or create it through the [Heroku Dashboard](https://dashboard.heroku.com/).

### 2. Add PostgreSQL Database

```bash
heroku addons:create heroku-postgresql:essential-0
```

This automatically sets the `DATABASE_URL` environment variable. The `essential-0` plan is the entry-level paid plan at $5/month.

### 3. Configure Environment Variables

Set all required environment variables:

```bash
# Django Secret Key (generate a new one for production)
heroku config:set SECRET_KEY="your-super-secret-key-here"

# Disable debug mode in production
heroku config:set DEBUG=False

# Set allowed hosts (replace with your Heroku app domain)
heroku config:set ALLOWED_HOSTS="your-app-name.herokuapp.com"

# Configure CSRF trusted origins (replace with your Heroku app domain)
heroku config:set CSRF_TRUSTED_ORIGINS="https://your-app-name.herokuapp.com"

# Facebook WhatsApp Business API (if needed)
heroku config:set FACEBOOK_TOKEN="your-facebook-access-token"
heroku config:set FACEBOOK_PHONE_NUMBER_ID="your-phone-number-id"
heroku config:set FACEBOOK_WEBHOOK_VERIFICATION="your-webhook-verification-token"
```

**Important**: Replace `your-app-name` with your actual Heroku app name.

### 4. Deploy from GitHub (Recommended)

#### Option A: Automatic Deployment from GitHub

1. Go to your Heroku app dashboard
2. Navigate to the **Deploy** tab
3. Under **Deployment method**, select **GitHub**
4. Connect your GitHub account if not already connected
5. Search for your repository `avictorino/wachat`
6. Click **Connect**
7. Enable **Automatic Deploys** from the `main` branch
8. Click **Deploy Branch** to trigger the initial deployment

Now, every push to the `main` branch will automatically deploy to Heroku.

#### Option B: Manual Deployment via Git

If you prefer to deploy manually:

```bash
# Add Heroku remote
heroku git:remote -a your-app-name

# Deploy the application
git push heroku main
```

### 5. Verify Database Configuration

The application is configured to automatically use Heroku's PostgreSQL through the `DATABASE_URL` environment variable. Verify it:

```bash
heroku config:get DATABASE_URL
```

### 6. Run Migrations

Migrations are automatically run during the release phase (defined in `Procfile`). However, you can manually run them if needed:

```bash
heroku run python manage.py migrate
```

### 7. Create a Superuser

```bash
heroku run python manage.py createsuperuser
```

### 8. Collect Static Files

Static files are collected automatically during deployment, but you can manually trigger it:

```bash
heroku run python manage.py collectstatic --noinput
```

## Configuration Files

### Procfile

Defines the processes that run on Heroku:

```
release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
web: gunicorn config.wsgi:application --workers 1 --threads 2 --timeout 60 --log-file -
```

- **release**: Runs migrations and collects static files automatically before each deployment
- **web**: Starts the web server using Gunicorn

### runtime.txt

Specifies the Python version:

```
python-3.12.3
```

### requirements.txt

Includes all Python dependencies, including:
- `gunicorn`: WSGI HTTP server for production
- `psycopg2-binary`: PostgreSQL adapter
- `dj-database-url`: Database configuration via URL
- `whitenoise`: Static file serving

## Monitoring and Logs

### View Logs

```bash
# Stream logs in real-time
heroku logs --tail

# View specific number of log lines
heroku logs -n 500
```

### Check Dyno Status

```bash
heroku ps
```

### Restart the Application

```bash
heroku restart
```

## Database Management

### Access PostgreSQL Console

```bash
heroku pg:psql
```

### Database Backups

```bash
# Create a backup
heroku pg:backups:capture

# View backups
heroku pg:backups

# Download a backup
heroku pg:backups:download
```

### Reset Database (⚠️ Destructive)

```bash
heroku pg:reset DATABASE_URL
heroku run python manage.py migrate
```

## Environment Variables Reference

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `SECRET_KEY` | Django secret key | Yes | `your-secret-key-here` |
| `DEBUG` | Debug mode (always False in production) | Yes | `False` |
| `ALLOWED_HOSTS` | Comma-separated list of allowed hosts | Yes | `your-app-name.herokuapp.com` |
| `CSRF_TRUSTED_ORIGINS` | Comma-separated CSRF trusted origins | Yes | `https://your-app-name.herokuapp.com` |
| `DATABASE_URL` | PostgreSQL connection URL | Auto-set | Heroku sets this automatically |
| `FACEBOOK_TOKEN` | Facebook API token | Optional | `your-token` |
| `FACEBOOK_PHONE_NUMBER_ID` | WhatsApp phone number ID | Optional | `your-phone-id` |
| `FACEBOOK_WEBHOOK_VERIFICATION` | Webhook verification token | Optional | `your-verification-token` |

## Production Security

The application automatically enables the following security features when `DEBUG=False`:

- SSL/HTTPS redirect (`SECURE_SSL_REDIRECT`)
- HTTP Strict Transport Security (HSTS) with 1 year duration
- Secure session cookies (`SESSION_COOKIE_SECURE`)
- Secure CSRF cookies (`CSRF_COOKIE_SECURE`)
- Proxy SSL header support for Heroku (`SECURE_PROXY_SSL_HEADER`)

## Scaling

### Scale Web Dynos

```bash
# Scale to multiple dynos
heroku ps:scale web=2

# Scale down to one dyno
heroku ps:scale web=1
```

### Upgrade Database

```bash
# View available plans
heroku addons:plans heroku-postgresql

# Upgrade to a larger plan
heroku addons:create heroku-postgresql:standard-0
```

## Troubleshooting

### Application Crashes

Check logs for errors:
```bash
heroku logs --tail
```

### Database Connection Issues

Verify DATABASE_URL is set:
```bash
heroku config:get DATABASE_URL
```

### Static Files Not Loading

Ensure WhiteNoise is installed and configured (already included in settings.py):
```bash
heroku run python manage.py collectstatic --noinput
```

### Migration Issues

Run migrations manually:
```bash
heroku run python manage.py migrate
```

## CI/CD Integration

With GitHub automatic deployment enabled:

1. Push changes to the `main` branch
2. Heroku automatically detects the push
3. Builds the application
4. Runs the release phase (migrations)
5. Deploys the new version
6. Restarts the web dynos

## Additional Resources

- [Heroku Python Documentation](https://devcenter.heroku.com/categories/python-support)
- [Heroku PostgreSQL Documentation](https://devcenter.heroku.com/categories/heroku-postgres)
- [Django Deployment Checklist](https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/)
- [Heroku CLI Reference](https://devcenter.heroku.com/articles/heroku-cli-commands)

## Support

For issues or questions:
- Check the [Heroku Status Page](https://status.heroku.com/)
- Review [Heroku DevCenter](https://devcenter.heroku.com/)
- Contact the project maintainers
