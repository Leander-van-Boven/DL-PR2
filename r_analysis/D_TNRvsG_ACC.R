library('ggplot2')
library('dplyr')

dig_none <- read.csv('./data/digits_dtl_none.csv', header=T)
dig_dig <- read.csv('./data/digits_dtl_same.csv', header=T)
dig_fash <- read.csv('./data/digits_dtl_fashion.csv', header=T)
fash_none <- read.csv('./data/fashion_dtl_none.csv', header=T)
fash_fash <- read.csv('./data/fashion_dtl_same.csv', header=T)
fash_dig <- read.csv('./data/fashion_dtl_digits.csv', header=T)

csv_data <- list(
  "Digits, no DTL" = dig_none,
  "Digits, DTL on digits" = dig_dig,
  "Digits, DTL on fashion" = dig_fash,
  "Fashion, no DTL" = fash_none,
  "Fashion, DTL on fashion" = fash_fash,
  "Fashion, DTL on digits" = fash_dig
)

data <- bind_rows(csv_data, .id="column_name")

ggplot(data, aes(x=epoch)) +
  geom_smooth(aes(y=d_acc, color='Discriminator accuracy'), method='loess', span=.1) +
  geom_smooth(aes(y=g_acc, color='Generator accuracy'), method='loess', span=.1) +
  theme(legend.position = 'bottom') +
  labs(x="Epoch",y="",colour="Metric") +
  facet_wrap(~ column_name)

ggsave("comparison.png", width=7, height=5)
