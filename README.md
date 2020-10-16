# MSQ optimization

## Config

To run this project you have to create a config file.  
In the project root folder:  
```sh
touch config.json
```

And insert to json the next data:
```json
{
  "DB_HOST": "xxx.xxx.xxx.xxx",
  "DB_DATABASE": "database",
  "DB_USER": "database_username",
  "DB_PASSWORD": "databasePassw0rd",
  "URL": "<api_endpoint>"
}
```

URL may look like:  
```
http(s)://<api_ip>:<api_port>/<api_version>/batch
Example:
http://opt-api.local:10012/v1/batch
```

## Run locally

### Test environment (flask)

```sh
bash build.sh
```

To test:  
```sh
$ curl http://localhost/
$
# empty reply, 204
```
### With cron

Change docker-compose file. Delete or comment next line:  
```sh
command: python main.py
```

And run:  
```sh
bash build.sh
```
