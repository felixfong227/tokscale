import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth/session";
import { listPersonalTokens } from "@/lib/auth/personalTokens";

export async function GET() {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
    }

    const tokens = await listPersonalTokens(session.id);

    return NextResponse.json({
      tokens: tokens.map((token) => ({
        id: token.id,
        name: token.name,
        createdAt: token.createdAt,
        lastUsedAt: token.lastUsedAt,
      })),
    });
  } catch (error) {
    console.error("Tokens list error:", error);
    return NextResponse.json(
      { error: "Failed to fetch tokens" },
      { status: 500 }
    );
  }
}
