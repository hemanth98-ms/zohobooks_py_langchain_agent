
from app.zoho_books import ZohoOAuthClient
from app.rag import RAGService
import time

def ingest_all():
    print("🚀 Starting Ingestion Process...")
    
    # Init generic clients
    client = ZohoOAuthClient()
    
    # Ensure token validity
    try:
        client.get_access_token()
    except RuntimeError:
        print("❌ Zoho Auth required. Please run the agent first to authenticate.")
        return

    rag = RAGService()
    
    page = 1
    has_more = True
    total_processed = 0

    while has_more:
        print(f"📄 Fetching page {page}...")
        try:
            # We fetch list then detail for each invoice to get line items which are crucial for search!
            # List usually doesn't have line items.
            res = client.list_invoices(page=page, per_page=25)
            invoices = res.get("invoices", [])
            
            if not invoices:
                has_more = False
                break
                
            for inv_summary in invoices:
                inv_id = inv_summary["invoice_id"]
                inv_number = inv_summary["invoice_number"]
                print(f"   🔹 Processing {inv_number} ({inv_id})...", end="", flush=True)
                
                try:
                    # Fetch Full Details (slow but necessary for good RAG)
                    detail_res = client.get_invoice(inv_id)
                    full_invoice = detail_res.get("invoice")
                    if full_invoice:
                        rag.add_invoice(full_invoice)
                        print(" ✅ Indexed")
                    else:
                        print(" ⚠️ No Detail Found")
                except Exception as e:
                     print(f" ❌ Error fetching detail: {e}")

                time.sleep(0.2) # Rate limit kindness
                total_processed += 1

            page_context = res.get("page_context", {})
            has_more = page_context.get("has_more_page", False)
            page += 1
            
        except Exception as e:
            print(f"❌ Error during list fetching: {e}")
            break

    print(f"\n✨ Ingestion Complete! Processed {total_processed} invoices.")

if __name__ == "__main__":
    ingest_all()
