"""
Tiny "knowledge base" for the demo customer-care agent.

In a real deployment these documents would come from your help center, PDFs,
Confluence, a database, etc., and you'd index them in a proper vector store.
Here we keep them as plain strings and let LangChain's TFIDFRetriever index
them in-memory so the demo runs with zero extra infrastructure or API keys.

Theme: "Ascent Automotive Group" — a fictional car dealership customer-care line.
"""

FAQ_DOCS = [
    {
        "topic": "financing",
        "content": (
            "Financing and loans: Ascent Automotive Group offers financing on new, used, and certified "
            "pre-owned vehicles. Customers can get pre-approved in minutes; APR depends on credit "
            "score, loan term, and down payment. A typical down payment is 10-20% of the vehicle "
            "price, and terms run from 36 to 72 months. We work with multiple lenders, and "
            "first-time buyers and trade-ins can lower the monthly payment. Bring a photo ID, proof "
            "of income, and proof of insurance to finalize a loan."
        ),
    },
    {
        "topic": "warranty",
        "content": (
            "Warranty and service plans: New vehicles include the manufacturer's warranty, "
            "typically 3 years or 36,000 miles bumper-to-bumper and 5 years or 60,000 miles on the "
            "powertrain. Certified pre-owned vehicles come with an extended limited warranty and a "
            "multi-point inspection. Used (non-certified) vehicles are sold as-is unless an extended "
            "service plan is purchased. Extended service contracts can be added at any time before "
            "the factory warranty expires."
        ),
    },
    {
        "topic": "trade_in",
        "content": (
            "Trade-ins: Ascent Automotive Group accepts trade-ins and applies the appraised value toward the "
            "price of your next vehicle, which can also reduce the sales tax you owe. To appraise a "
            "trade-in we need the make, model, year, mileage, and overall condition. Bring the title "
            "or loan payoff information and both key fobs. A same-day written offer is provided and "
            "is valid for 7 days."
        ),
    },
    {
        "topic": "inventory",
        "content": (
            "Checking inventory / availability: To check whether a specific vehicle is available we "
            "need the make and model, and optionally the year, budget, and whether the customer "
            "wants new, used, or certified pre-owned. As a rule of thumb, new models start around "
            "$28,000, certified pre-owned around $21,000, and used around $16,000. If the customer's "
            "budget is below the starting price we suggest used options or financing to bridge the "
            "gap, and we can always book a test drive for an in-stock vehicle."
        ),
    },
    {
        "topic": "test_drive_and_service",
        "content": (
            "Booking a test drive or service: Customers can schedule a test drive for any in-stock "
            "vehicle, or a service/maintenance appointment for a car they own. We collect the make "
            "and model, a callback phone number, and a preferred time, and provide an appointment "
            "reference number (format APT-XXXXXX). A valid driver's license is required for a test "
            "drive. Service appointments cover maintenance, recalls, and repairs."
        ),
    },
    {
        "topic": "returns_and_delivery",
        "content": (
            "Return policy, hours, and delivery: Ascent Automotive Group offers a 7-day or 500-mile "
            "money-back guarantee on most vehicles so customers can buy with confidence. The "
            "showroom is open Monday to Saturday, 9am to 7pm, and closed Sunday. Home delivery is "
            "available within 50 miles for a flat fee, and out-of-state buyers can arrange shipping."
        ),
    },
    {
        "topic": "off_scope",
        "content": (
            "Off-scope topics: For questions about a specific lienholder payoff, legal/registration "
            "paperwork for a particular state, or complex insurance claims, the customer should be "
            "directed to a human sales or finance representative, as those are outside the scope of "
            "the automated customer-care assistant."
        ),
    },
]

# Flattened text list used to build the retriever.
FAQ_TEXTS = [d["content"] for d in FAQ_DOCS]
FAQ_METADATAS = [{"topic": d["topic"]} for d in FAQ_DOCS]
