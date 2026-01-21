import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from supabase import create_client, Client

def main():
    # Load environment variables
    load_dotenv()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set in .env or environment variables.")
        # We don't exit here to allow user to see the error and maybe fix it in the same session if possible, 
        # but in this script execution it's fatal.
        sys.exit(1)

    try:
        supabase: Client = create_client(url, key)

        # Calculate date 24 months ago
        # SQL: date >= (current_date - interval '24 months')
        start_date = (datetime.now() - relativedelta(months=24)).strftime('%Y-%m-%d')
        print(f"Querying data since: {start_date}")

        # Execute Query
        response = supabase.table('macro_regime') \
            .select('date, regime_id, regime_label, stress_flag') \
            .gte('date', start_date) \
            .order('date', desc=False) \
            .execute()

        data = response.data

        if not data:
            print("No data found.")
            return

        # Print results in a tabular format
        print(f"{'date':<12} {'regime_id':<10} {'regime_label':<20} {'stress_flag'}")
        print("-" * 55)
        for row in data:
            date_val = row.get('date', '')
            r_id = str(row.get('regime_id', ''))
            r_lbl = row.get('regime_label', '')
            s_flag = str(row.get('stress_flag', ''))
            print(f"{date_val:<12} {r_id:<10} {r_lbl:<20} {s_flag}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
