FROM python:3.11

RUN apt-get update && apt-get install -y nodejs npm

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /root/.mcp

RUN echo '{ \
  "local-strapi": { \
    "api_url": "http://strapi:1337", \
    "api_key": "7111d6b882cbcfc25eb1d8a68e1b2cab31cc4bab5270a534cb28618cb95c435537935518fded96823de0603ae6fcf0587f1b98f110240d2e7b9fa6798b836cff0a7c0af4c1f3d144df01a67173f5387f23bb38bbe7ccd24028cff18cb399edd7a4a68e4876970acc97c9e4894ade83fed9a490c66a5e09fc27b6c6484f902f4c", \
    "version": "5.*" \
  } \
}' > /root/.mcp/strapi-mcp-server.config.json

CMD ["tail", "-f", "/dev/null"]