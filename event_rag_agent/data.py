"""Event data for the RAG system."""

from langchain_core.documents import Document

# 20 event records
EVENTS = [
    Document(
        page_content="Tech Conference 2025 is a major technology conference happening on March 15, 2025. The event will be held at Mumbai Convention Center in Mumbai. Ticket prices start at ₹2500 for general admission. This conference features keynote speakers from top tech companies discussing AI, cloud computing, and emerging technologies.",
        metadata={"event_name": "Tech Conference 2025", "date": "2025-03-15", "venue": "Mumbai Convention Center", "city": "Mumbai", "price": 2500, "category": "Technology"}
    ),
    Document(
        page_content="Bollywood Music Night is scheduled for April 22, 2025 at Phoenix Marketcity Arena in Pune. Tickets are priced at ₹1500. Experience an evening of live Bollywood music performances featuring popular playback singers and bands.",
        metadata={"event_name": "Bollywood Music Night", "date": "2025-04-22", "venue": "Phoenix Marketcity Arena", "city": "Pune", "price": 1500, "category": "Music"}
    ),
    Document(
        page_content="Startup Summit India will take place on May 10, 2025 at Bangalore International Exhibition Centre in Bangalore. Entry fee is ₹3000 per person. This summit brings together entrepreneurs, investors, and innovators to discuss startup ecosystem trends and funding opportunities.",
        metadata={"event_name": "Startup Summit India", "date": "2025-05-10", "venue": "Bangalore International Exhibition Centre", "city": "Bangalore", "price": 3000, "category": "Business"}
    ),
    Document(
        page_content="Classical Dance Festival is happening on June 5, 2025 at Nehru Centre Auditorium in Mumbai. Tickets cost ₹800. Watch renowned classical dancers perform Bharatanatyam, Kathak, and Odissi in this cultural celebration.",
        metadata={"event_name": "Classical Dance Festival", "date": "2025-06-05", "venue": "Nehru Centre Auditorium", "city": "Mumbai", "price": 800, "category": "Arts"}
    ),
    Document(
        page_content="Food and Wine Expo 2025 is scheduled for July 18, 2025 at Delhi Haat in Delhi. Entry tickets are ₹500. Explore cuisines from around the world, attend cooking demonstrations, and participate in wine tasting sessions.",
        metadata={"event_name": "Food and Wine Expo 2025", "date": "2025-07-18", "venue": "Delhi Haat", "city": "Delhi", "price": 500, "category": "Food"}
    ),
    Document(
        page_content="AI and Machine Learning Workshop will be conducted on August 12, 2025 at Hyderabad International Convention Centre in Hyderabad. Registration fee is ₹4500. This hands-on workshop covers deep learning, NLP, and computer vision with practical coding sessions.",
        metadata={"event_name": "AI and Machine Learning Workshop", "date": "2025-08-12", "venue": "Hyderabad International Convention Centre", "city": "Hyderabad", "price": 4500, "category": "Technology"}
    ),
    Document(
        page_content="Rock Concert 2025 featuring international rock bands is on September 20, 2025 at DY Patil Stadium in Mumbai. Tickets range from ₹2000 to ₹8000 depending on seating. Get ready for an electrifying night of live rock music.",
        metadata={"event_name": "Rock Concert 2025", "date": "2025-09-20", "venue": "DY Patil Stadium", "city": "Mumbai", "price": 2000, "category": "Music"}
    ),
    Document(
        page_content="Digital Marketing Masterclass is on October 8, 2025 at The Lalit Hotel in Bangalore. Participation fee is ₹3500. Learn SEO, social media marketing, content strategy, and analytics from industry experts.",
        metadata={"event_name": "Digital Marketing Masterclass", "date": "2025-10-08", "venue": "The Lalit Hotel", "city": "Bangalore", "price": 3500, "category": "Business"}
    ),
    Document(
        page_content="Diwali Mela 2025 is a festive celebration on October 25, 2025 at Ramoji Film City in Hyderabad. Free entry for all. Enjoy traditional performances, food stalls, shopping, and fireworks display.",
        metadata={"event_name": "Diwali Mela 2025", "date": "2025-10-25", "venue": "Ramoji Film City", "city": "Hyderabad", "price": 0, "category": "Festival"}
    ),
    Document(
        page_content="Photography Exhibition 2025 showcasing works by emerging photographers will be held on November 15, 2025 at National Gallery of Modern Art in Delhi. Entry is ₹200. Explore diverse photographic styles from landscape to portrait.",
        metadata={"event_name": "Photography Exhibition 2025", "date": "2025-11-15", "venue": "National Gallery of Modern Art", "city": "Delhi", "price": 200, "category": "Arts"}
    ),
    Document(
        page_content="Marathon 2025 for fitness enthusiasts is scheduled for December 1, 2025 at Marine Drive in Mumbai. Registration costs ₹1000. Choose from 5K, 10K, or half marathon distances and support a charitable cause.",
        metadata={"event_name": "Marathon 2025", "date": "2025-12-01", "venue": "Marine Drive", "city": "Mumbai", "price": 1000, "category": "Sports"}
    ),
    Document(
        page_content="New Year Gala 2026 celebration will take place on December 31, 2025 at JW Marriott in Pune. Tickets are ₹5000 per person. Ring in the new year with live music, DJ, gourmet dinner, and champagne.",
        metadata={"event_name": "New Year Gala 2026", "date": "2025-12-31", "venue": "JW Marriott", "city": "Pune", "price": 5000, "category": "Party"}
    ),
    Document(
        page_content="Yoga and Wellness Retreat is a 3-day event from January 10-12, 2026 at Atmantan Wellness Resort in Pune. Package price is ₹15000. Includes yoga sessions, meditation, spa treatments, and healthy meals.",
        metadata={"event_name": "Yoga and Wellness Retreat", "date": "2026-01-10", "venue": "Atmantan Wellness Resort", "city": "Pune", "price": 15000, "category": "Wellness"}
    ),
    Document(
        page_content="Comic Con India 2026 for comic and pop culture fans is on February 14, 2026 at NSCI Dome in Mumbai. Tickets cost ₹1200. Meet artists, cosplay, attend panels, and explore comics, anime, and gaming.",
        metadata={"event_name": "Comic Con India 2026", "date": "2026-02-14", "venue": "NSCI Dome", "city": "Mumbai", "price": 1200, "category": "Entertainment"}
    ),
    Document(
        page_content="Blockchain and Crypto Summit will be held on March 5, 2026 at Hotel Taj Palace in Delhi. Entry fee is ₹4000. Discuss cryptocurrency trends, DeFi, NFTs, and blockchain technology with industry leaders.",
        metadata={"event_name": "Blockchain and Crypto Summit", "date": "2026-03-05", "venue": "Hotel Taj Palace", "city": "Delhi", "price": 4000, "category": "Technology"}
    ),
    Document(
        page_content="Stand-up Comedy Night featuring popular comedians is on March 28, 2026 at Phoenix Marketcity in Bangalore. Tickets are ₹800. Laugh out loud with hilarious performances from top stand-up comedians.",
        metadata={"event_name": "Stand-up Comedy Night", "date": "2026-03-28", "venue": "Phoenix Marketcity", "city": "Bangalore", "price": 800, "category": "Entertainment"}
    ),
    Document(
        page_content="Fashion Week 2026 showcasing designer collections is scheduled for April 15-17, 2026 at Jawaharlal Nehru Stadium in Delhi. Passes start at ₹2500. Watch runway shows, meet designers, and explore latest fashion trends.",
        metadata={"event_name": "Fashion Week 2026", "date": "2026-04-15", "venue": "Jawaharlal Nehru Stadium", "city": "Delhi", "price": 2500, "category": "Fashion"}
    ),
    Document(
        page_content="Educational Career Fair 2026 for students is on May 20, 2026 at World Trade Center in Mumbai. Free entry for students. Explore educational programs, universities, and career opportunities with guidance counselors.",
        metadata={"event_name": "Educational Career Fair 2026", "date": "2026-05-20", "venue": "World Trade Center", "city": "Mumbai", "price": 0, "category": "Education"}
    ),
    Document(
        page_content="Film Festival 2026 screening independent films is happening June 8-10, 2026 at Siri Fort Auditorium in Delhi. Day pass costs ₹600. Watch award-winning independent films, documentaries, and short films from around the world.",
        metadata={"event_name": "Film Festival 2026", "date": "2026-06-08", "venue": "Siri Fort Auditorium", "city": "Delhi", "price": 600, "category": "Arts"}
    ),
    Document(
        page_content="Gaming Championship 2026 for esports enthusiasts will be held on July 25, 2026 at Bangalore Palace Grounds in Bangalore. Entry is ₹1500. Compete or watch professional gamers in tournaments across multiple games.",
        metadata={"event_name": "Gaming Championship 2026", "date": "2026-07-25", "venue": "Bangalore Palace Grounds", "city": "Bangalore", "price": 1500, "category": "Gaming"}
    )
]
