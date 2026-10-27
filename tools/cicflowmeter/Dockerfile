# CICFlowMeter, fixed version of Engelen et al. (2021), with two small patches so
# that new extractions match the label-corrected CIC-IDS2017 release exactly:
#   1. timestamps carry the AM/PM marker ("04/07/2017 01:53:44 PM"); the stock
#      tool prints "hh" (12-hour) without it, so 01:53 and 13:53 are ambiguous;
#   2. timestamps are always written in UTC, whatever the host time zone.
# Build context = a clone of https://github.com/GintsEngelen/CICFlowMeter
# (scripts/windows/01_build_cicflowmeter.ps1 does this for you).

FROM alpine:3.18 AS builder
RUN apk add --update --no-cache unzip openjdk8 libpcap-dev
COPY . /CICFlowMeter/
WORKDIR /CICFlowMeter
RUN sed -i 's/new SimpleDateFormat(format);/new SimpleDateFormat(format, java.util.Locale.US); simpleFormatter.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));/' \
        src/main/java/cic/cs/unb/ca/jnetpcap/DateFormatter.java \
 && sed -i 's#"dd/MM/yyyy hh:mm:ss")#"dd/MM/yyyy hh:mm:ss a")#g' \
        src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java \
 && grep -q 'getTimeZone("UTC")' src/main/java/cic/cs/unb/ca/jnetpcap/DateFormatter.java \
 && test "$(grep -c 'hh:mm:ss a"' src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java)" = "2" \
 && tr -d '\r' < gradlew > gradlew.unix && mv gradlew.unix gradlew && chmod +x gradlew
RUN ./gradlew --no-daemon distZip
RUN UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o build/distributions/CICFlowMeter-*.zip -d build/distributions

FROM alpine:3.18
RUN apk add --update --no-cache openjdk8 libpcap-dev
WORKDIR /CICFlowMeter
COPY --from=builder /CICFlowMeter/build/distributions/CICFlowMeter-* .
# The cfm start script uses a relative library path (../lib/native).
WORKDIR /CICFlowMeter/bin
ENV TZ=UTC JAVA_OPTS="-Xmx4g"
# usage: docker run -v <pcap folder>:/in -v <csv folder>:/out cicflowmeter-utc /in /out
ENTRYPOINT ["./cfm"]
