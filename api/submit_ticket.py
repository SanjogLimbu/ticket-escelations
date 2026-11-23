from fastapi import APIRouter

##

router = APIRouter()

##

@router.post("/submit_ticket")
def health():
    return {"status": "ok"}
