FROM node:20-alpine

WORKDIR /app

# Install OS-level deps required by some npm packages (sharp etc.).
RUN apk add --no-cache libc6-compat

COPY package.json ./
COPY yarn.lock* ./

RUN if [ -f yarn.lock ]; then \
        yarn install --frozen-lockfile; \
    else \
        yarn install; \
    fi

COPY . ./

EXPOSE 3000

CMD ["yarn", "dev", "--hostname", "0.0.0.0"]
