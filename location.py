import phonenumbers
from phonenumbers import geocoder, carrier
phone_number1=phonenumbers.parse("+33758212812")

print("\nphone Numbers location:")
print(geocoder.description_for_number(phone_number1,"en"))
print("\ncarrier:")
print(carrier.name_for_number(phone_number1,"en"))
from phonenumbers import timezone
time_zones=timezone.time_zones_for_number(phone_number1)
print("\nTime Zone(s):")
print(",".join(time_zones))