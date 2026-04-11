SELECT
    c.household_key,
    c.campaign,
    cd.campaign_type,
    cd.start_day,
    cd.end_day,
    cd.start_period,
    cd.end_period
FROM {{ source('raw', 'campaign_table') }} c
INNER JOIN {{ ref('stg_campaigns') }} cd ON c.campaign = cd.campaign
